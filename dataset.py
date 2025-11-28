import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


class StarRemovalDataset(Dataset):
    """
    Dataset for star removal training
    Assumes structure:
    - train/input/*.png (or jpg)
    - train/target/*.png (or jpg)
    - val/input/*.png (or jpg)
    - val/target/*.png (or jpg)
    
    Files have matching names between input and target
    """
    def __init__(self, root_dir, split='train', image_size=None):
        """
        Args:
            root_dir: Path to dataset root
            split: 'train' or 'val'
            image_size: Optional, resize images to this size (H, W). None to keep original
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        
        # Get input and target directories
        self.input_dir = self.root_dir / split / 'input'
        self.target_dir = self.root_dir / split / 'target'
        
        # Find all input images
        self.input_images = sorted(list(self.input_dir.glob('*.png')) + 
                                   list(self.input_dir.glob('*.jpg')) +
                                   list(self.input_dir.glob('*.jpeg')))
        
        if len(self.input_images) == 0:
            raise ValueError(f"No images found in {self.input_dir}")
        
        # Verify matching target images exist
        self._verify_dataset()
        
        print(f"Loaded {len(self.input_images)} image pairs for {split}")
        
    def _verify_dataset(self):
        """Verify that each input has a matching target"""
        missing = []
        for input_path in self.input_images:
            target_path = self.target_dir / input_path.name
            if not target_path.exists():
                missing.append(input_path.name)
        
        if missing:
            raise ValueError(f"Missing target images for: {missing[:10]}...")
    
    def __len__(self):
        return len(self.input_images)
    
    def __getitem__(self, idx):
        # Load input and target images
        input_path = self.input_images[idx]
        target_path = self.target_dir / input_path.name
        
        # Load as RGB
        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')
        
        # Resize if specified
        if self.image_size is not None:
            input_img = input_img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
            target_img = target_img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        
        # Convert to tensor and normalize to [0, 1]
        input_tensor = transforms.ToTensor()(input_img)
        target_tensor = transforms.ToTensor()(target_img)
        
        return {
            'input': input_tensor,
            'target': target_tensor,
            'filename': input_path.name
        }


def create_dataloaders(root_dir, batch_size=8, num_workers=4, image_size=None, pin_memory=True):
    """
    Create train and validation dataloaders
    
    Args:
        root_dir: Path to dataset root
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        image_size: Optional (H, W) to resize images
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        train_loader, val_loader
    """
    train_dataset = StarRemovalDataset(root_dir, split='train', image_size=image_size)
    val_dataset = StarRemovalDataset(root_dir, split='val', image_size=image_size)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch for stable training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    dataset = StarRemovalDataset('.', split='train')
    print(f"Dataset size: {len(dataset)}")
    
    # Test sample
    sample = dataset[0]
    print(f"Input shape: {sample['input'].shape}")
    print(f"Target shape: {sample['target'].shape}")
    print(f"Filename: {sample['filename']}")
    print(f"Input range: [{sample['input'].min():.3f}, {sample['input'].max():.3f}]")
    print(f"Target range: [{sample['target'].min():.3f}, {sample['target'].max():.3f}]")
