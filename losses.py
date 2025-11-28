import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features
    Essential for texture coherence - prevents blob artifacts
    """
    def __init__(self, layers=['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4']):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features.eval()
        
        # Freeze VGG parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        self.vgg = vgg
        self.layer_name_mapping = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_4': 17,
            'relu4_4': 26,
            'relu5_4': 35
        }
        self.layers = [self.layer_name_mapping[name] for name in layers]
        
    def forward(self, pred, target):
        """
        Args:
            pred: predicted starless image
            target: ground truth starless image
        """
        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred = (pred - mean) / std
        target = (target - mean) / std
        
        loss = 0.0
        x_pred = pred
        x_target = target
        
        for i, layer in enumerate(self.vgg):
            x_pred = layer(x_pred)
            x_target = layer(x_target)
            
            if i in self.layers:
                loss += F.l1_loss(x_pred, x_target)
        
        return loss


class TextureLoss(nn.Module):
    """
    Texture loss based on Gram matrix
    Ensures realistic texture reconstruction in star regions
    """
    def __init__(self):
        super().__init__()
        
    def gram_matrix(self, x):
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    def forward(self, pred, target):
        """Compute texture similarity via Gram matrices"""
        gram_pred = self.gram_matrix(pred)
        gram_target = self.gram_matrix(target)
        return F.mse_loss(gram_pred, gram_target)


class FrequencyLoss(nn.Module):
    """
    Frequency domain loss
    Helps preserve fine details and high-frequency textures
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        """Compare in frequency domain"""
        # 2D FFT
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)
        
        # Magnitude spectrum
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        return F.l1_loss(pred_mag, target_mag)


class ColorConsistencyLoss(nn.Module):
    """
    Color Consistency Loss
    
    Stabilizes chromatic rendering across tiles.
    Prevents:
    - Color tint differences
    - Micro color halos
    - Posterization
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        """Compare global color statistics"""
        mean_pred = pred.mean(dim=[2, 3])
        mean_target = target.mean(dim=[2, 3])
        return F.l1_loss(mean_pred, mean_target)


class LocalVarianceLoss(nn.Module):
    """
    Local Variance Regularizer
    
    Eliminates artificial patches/spots in dark regions.
    Maintains fine texture while removing large-scale artifacts.
    """
    def __init__(self, patch_size=5):
        super().__init__()
        self.patch_size = patch_size
        
    def forward(self, pred, target):
        """Compute local variance of prediction error"""
        diff = pred - target
        var = F.avg_pool2d(diff**2, self.patch_size, stride=1, padding=self.patch_size//2)
        return var.mean()


def star_penalty_loss(residual):
    """
    Star Penalty Term
    
    Prevents over-removal of:
    - Comets
    - Semi-faint stars
    - Compact galaxies
    
    Penalizes residuals that are too strong outside stellar cores.
    """
    # Penalize negative residuals (over-removal)
    return torch.mean(torch.relu(-residual))


class CombinedLoss(nn.Module):
    """
    Combined loss for realistic star removal
    
    - L1: Pixel-wise accuracy
    - Perceptual: Texture coherence (anti-blob)
    - Texture: Realistic texture synthesis
    - Frequency: Detail preservation
    - Color Consistency: Chromatic stability across tiles
    - Local Variance: Anti-patch/anti-spot regularization
    - Star Penalty: Prevents over-removal of comets/galaxies
    
    This combination ensures:
    1. Fast learning (L1 provides strong gradient)
    2. Realistic textures (perceptual + texture)
    3. No synthetic blobs (perceptual)
    4. Detail preservation (frequency)
    5. Color stability (color consistency)
    6. Clean backgrounds (local variance)
    7. Safe removal (star penalty)
    """
    def __init__(self, 
                 lambda_l1=1.0,
                 lambda_perceptual=0.1,
                 lambda_texture=0.05,
                 lambda_freq=0.01,
                 lambda_color=0.05,
                 lambda_variance=0.05,
                 lambda_penalty=0.01):
        super().__init__()
        
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_texture = lambda_texture
        self.lambda_freq = lambda_freq
        self.lambda_color = lambda_color
        self.lambda_variance = lambda_variance
        self.lambda_penalty = lambda_penalty
        
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.texture_loss = TextureLoss()
        self.freq_loss = FrequencyLoss()
        self.color_loss = ColorConsistencyLoss()
        self.variance_loss = LocalVarianceLoss()
        
    def forward(self, pred, target, residual=None):
        """
        Args:
            pred: predicted starless image [B, 3, H, W]
            target: ground truth starless image [B, 3, H, W]
            residual: optional residual output for penalty term [B, 3, H, W]
        
        Returns:
            total_loss, loss_dict
        """
        # L1 loss - fast learning
        l1 = self.l1_loss(pred, target)
        
        # Perceptual loss - texture coherence
        perceptual = self.perceptual_loss(pred, target)
        
        # Texture loss - realistic synthesis
        texture = self.texture_loss(pred, target)
        
        # Frequency loss - detail preservation
        freq = self.freq_loss(pred, target)
        
        # Color consistency - chromatic stability
        color = self.color_loss(pred, target)
        
        # Local variance - anti-patch regularization
        variance = self.variance_loss(pred, target)
        
        # Star penalty - prevent over-removal
        penalty = 0.0
        if residual is not None:
            penalty = star_penalty_loss(residual)
        
        # Combine
        total = (self.lambda_l1 * l1 + 
                self.lambda_perceptual * perceptual +
                self.lambda_texture * texture +
                self.lambda_freq * freq +
                self.lambda_color * color +
                self.lambda_variance * variance +
                self.lambda_penalty * penalty)
        
        loss_dict = {
            'total': total.item(),
            'l1': l1.item(),
            'perceptual': perceptual.item(),
            'texture': texture.item(),
            'frequency': freq.item(),
            'color': color.item(),
            'variance': variance.item(),
            'penalty': penalty.item() if isinstance(penalty, torch.Tensor) else penalty
        }
        
        return total, loss_dict


if __name__ == "__main__":
    # Test losses
    pred = torch.randn(2, 3, 256, 256)
    target = torch.randn(2, 3, 256, 256)
    residual = torch.randn(2, 3, 256, 256)
    
    criterion = CombinedLoss()
    loss, loss_dict = criterion(pred, target, residual)
    
    print("Loss components:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")
