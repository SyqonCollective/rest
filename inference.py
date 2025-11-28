import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
import argparse

from model import StarRemovalNet


def hann_window_2d(h, w):
    """
    Create 2D Hann window for boundary-aware tile blending
    
    Prevents:
    - Visible seams between tiles
    - Structural differences at tile boundaries
    - Micro-stitching artifacts
    """
    hann_h = torch.hann_window(h, periodic=False)
    hann_w = torch.hann_window(w, periodic=False)
    window_2d = hann_h.unsqueeze(1) * hann_w.unsqueeze(0)
    return window_2d


class StarRemover:
    """Inference class for star removal"""
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = StarRemovalNet(in_channels=3, base_channels=64, num_blocks=6)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded model from {checkpoint_path}")
        print(f"Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
        
    @torch.no_grad()
    def remove_stars(self, input_image_path, output_path=None, tile_size=None, overlap=64):
        """
        Remove stars from an image
        
        Args:
            input_image_path: Path to input image
            output_path: Path to save starless image (optional)
            tile_size: Process in tiles of this size (for large images). None = full image
            overlap: Overlap between tiles in pixels (for smooth blending)
        
        Returns:
            starless_image: PIL Image
        """
        # Load image
        input_img = Image.open(input_image_path).convert('RGB')
        original_size = input_img.size
        
        # Convert to tensor
        transform = transforms.ToTensor()
        input_tensor = transform(input_img).unsqueeze(0).to(self.device)
        
        # Process full image or with tiling
        if tile_size is None or (input_tensor.shape[2] <= tile_size and input_tensor.shape[3] <= tile_size):
            # Process full image
            with torch.cuda.amp.autocast():
                output_tensor = self.model(input_tensor)
        else:
            # Process with tiling and Hann window blending
            output_tensor = self._process_with_tiles(input_tensor, tile_size, overlap)
        
        # Convert back to PIL image
        output_tensor = output_tensor.squeeze(0).cpu()
        output_tensor = torch.clamp(output_tensor, 0, 1)
        
        # Convert to numpy
        output_np = output_tensor.permute(1, 2, 0).numpy()
        output_np = (output_np * 255).astype(np.uint8)
        
        starless_image = Image.fromarray(output_np)
        
        # Save if path provided
        if output_path is not None:
            starless_image.save(output_path)
            print(f"Saved starless image to {output_path}")
        
        return starless_image
    
    @torch.no_grad()
    def _process_with_tiles(self, input_tensor, tile_size, overlap):
        """
        Process large image with overlapping tiles and Hann window blending
        
        Ensures seamless tile fusion with no visible boundaries.
        """
        b, c, h, w = input_tensor.shape
        stride = tile_size - overlap
        
        # Create output accumulator
        output_acc = torch.zeros_like(input_tensor)
        weight_acc = torch.zeros((1, 1, h, w), device=input_tensor.device)
        
        # Create Hann window for blending
        hann = hann_window_2d(tile_size, tile_size).to(input_tensor.device)
        hann = hann.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Process tiles
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # Get tile boundaries
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                
                # Extract tile
                tile = input_tensor[:, :, y:y_end, x:x_end]
                
                # Pad if needed
                pad_h = tile_size - tile.shape[2]
                pad_w = tile_size - tile.shape[3]
                if pad_h > 0 or pad_w > 0:
                    tile = torch.nn.functional.pad(tile, (0, pad_w, 0, pad_h), mode='reflect')
                
                # Process tile
                with torch.cuda.amp.autocast():
                    tile_output = self.model(tile)
                
                # Remove padding
                if pad_h > 0 or pad_w > 0:
                    tile_output = tile_output[:, :, :tile_size-pad_h, :tile_size-pad_w]
                
                # Get window for this tile
                tile_h, tile_w = tile_output.shape[2], tile_output.shape[3]
                window = hann[:, :, :tile_h, :tile_w]
                
                # Accumulate with weighted blending
                output_acc[:, :, y:y_end, x:x_end] += tile_output * window
                weight_acc[:, :, y:y_end, x:x_end] += window
        
        # Normalize by accumulated weights
        output_tensor = output_acc / (weight_acc + 1e-8)
        
        return output_tensor
    
    @torch.no_grad()
    def batch_process(self, input_dir, output_dir):
        """
        Process all images in a directory
        
        Args:
            input_dir: Directory with input images
            output_dir: Directory to save starless images
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Find all images
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(ext))
            image_files.extend(input_path.glob(ext.upper()))
        
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        for img_file in image_files:
            print(f"Processing {img_file.name}...")
            output_file = output_path / img_file.name
            self.remove_stars(img_file, output_file)
        
        print(f"\nBatch processing complete! Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Star Removal Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image or directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path or directory')
    parser.add_argument('--batch', action='store_true',
                       help='Batch process directory')
    parser.add_argument('--tile-size', type=int, default=None,
                       help='Tile size for large images (None = full image)')
    parser.add_argument('--overlap', type=int, default=64,
                       help='Overlap between tiles for smooth blending')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Create remover
    remover = StarRemover(args.checkpoint, device=args.device)
    
    # Process
    if args.batch:
        remover.batch_process(args.input, args.output)
    else:
        remover.remove_stars(args.input, args.output, tile_size=args.tile_size, overlap=args.overlap)


if __name__ == "__main__":
    main()
