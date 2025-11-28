import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import time
from tqdm import tqdm
import json

from model import StarRemovalNet, count_parameters
from dataset import create_dataloaders
from losses import CombinedLoss


class Trainer:
    """
    Optimized trainer for RTX 5090
    - Mixed precision (FP16) for 2x speedup
    - Gradient accumulation for effective large batches
    - Fast learning from first epochs
    """
    def __init__(self, 
                 model,
                 train_loader,
                 val_loader,
                 device='cuda',
                 lr=2e-4,
                 epochs=100,
                 checkpoint_dir='checkpoints',
                 log_interval=100,
                 gradient_accumulation_steps=1):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_interval = log_interval
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Loss function - optimized for fast learning + realistic textures
        self.criterion = CombinedLoss(
            lambda_l1=1.0,          # Strong pixel-wise gradient
            lambda_perceptual=0.1,  # Texture coherence
            lambda_texture=0.05,    # Realistic synthesis
            lambda_freq=0.01,       # Detail preservation
            lambda_color=0.05,      # Chromatic stability
            lambda_variance=0.05,   # Anti-patch regularization
            lambda_penalty=0.01     # Prevent over-removal
        ).to(device)
        
        # Optimizer - AdamW with cosine annealing
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=1e-4
        )
        
        # Learning rate scheduler - cosine annealing with warmup
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=1e-6
        )
        
        # Mixed precision scaler for RTX 5090
        self.scaler = GradScaler()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        print(f"Trainer initialized:")
        print(f"  Model parameters: {count_parameters(model):,}")
        print(f"  Device: {device}")
        print(f"  Training samples: {len(train_loader.dataset)}")
        print(f"  Validation samples: {len(val_loader.dataset)}")
        print(f"  Batch size: {train_loader.batch_size}")
        print(f"  Gradient accumulation: {gradient_accumulation_steps}")
        print(f"  Effective batch size: {train_loader.batch_size * gradient_accumulation_steps}")
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_losses = {'total': 0, 'l1': 0, 'perceptual': 0, 'texture': 0, 'frequency': 0}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Mixed precision forward pass
            with autocast():
                outputs = self.model(inputs)
                
                # Compute residual for penalty term
                residual = inputs - outputs
                
                loss, loss_dict = self.criterion(outputs, targets, residual)
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping for stability
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            # Track losses
            epoch_loss += loss_dict['total']
            for k in epoch_losses.keys():
                epoch_losses[k] += loss_dict[k]
            
            # Update progress bar
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{loss_dict['total']:.4f}",
                    'l1': f"{loss_dict['l1']:.4f}",
                    'perc': f"{loss_dict['perceptual']:.4f}"
                })
        
        # Average losses
        num_batches = len(self.train_loader)
        avg_loss = epoch_loss / num_batches
        for k in epoch_losses.keys():
            epoch_losses[k] /= num_batches
        
        return avg_loss, epoch_losses
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        val_losses = {'total': 0, 'l1': 0, 'perceptual': 0, 'texture': 0, 'frequency': 0}
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            
            with autocast():
                outputs = self.model(inputs)
                
                # Compute residual for penalty term
                residual = inputs - outputs
                
                loss, loss_dict = self.criterion(outputs, targets, residual)
            
            val_loss += loss_dict['total']
            for k in val_losses.keys():
                val_losses[k] += loss_dict[k]
        
        # Average losses
        num_batches = len(self.val_loader)
        avg_loss = val_loss / num_batches
        for k in val_losses.keys():
            val_losses[k] /= num_batches
        
        return avg_loss, val_losses
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        # Save latest
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"  → Saved best model (val_loss: {self.best_val_loss:.4f})")
        
        # Save periodic checkpoint every 10 epochs
        if (self.current_epoch + 1) % 10 == 0:
            epoch_path = self.checkpoint_dir / f'epoch_{self.current_epoch+1}.pth'
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self):
        """Full training loop"""
        print("\n" + "="*60)
        print("Starting training")
        print("="*60 + "\n")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_loss, train_losses = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_losses = self.validate()
            self.val_losses.append(val_loss)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log results
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1}/{self.epochs} - {epoch_time:.1f}s")
            print(f"  LR: {current_lr:.6f}")
            print(f"  Train Loss: {train_loss:.4f} (L1: {train_losses['l1']:.4f}, Perc: {train_losses['perceptual']:.4f})")
            print(f"  Val Loss:   {val_loss:.4f} (L1: {val_losses['l1']:.4f}, Perc: {val_losses['perceptual']:.4f})")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(is_best=is_best)
        
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print(f"Training complete! Total time: {total_time/3600:.2f}h")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("="*60 + "\n")
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        with open(self.checkpoint_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)


def main():
    """Main training script"""
    # Configuration
    config = {
        'data_root': '.',
        'batch_size': 16,           # RTX 5090 can handle large batches
        'num_workers': 8,           # Adjust based on CPU cores
        'image_size': None,         # None = use original size, or (H, W) to resize
        'base_channels': 64,        # Model width
        'num_blocks': 6,            # Model depth (6-8 optimal for balance)
        'lr': 2e-4,                 # Learning rate
        'epochs': 100,
        'gradient_accumulation_steps': 1,  # Increase if OOM
        'checkpoint_dir': 'checkpoints',
        'resume': None              # Path to checkpoint to resume from
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Create dataloaders
    print("\nLoading dataset...")
    train_loader, val_loader = create_dataloaders(
        root_dir=config['data_root'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        image_size=config['image_size'],
        pin_memory=True
    )
    
    # Create model
    print("\nCreating model...")
    model = StarRemovalNet(
        in_channels=3,
        base_channels=config['base_channels'],
        num_blocks=config['num_blocks']
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=config['lr'],
        epochs=config['epochs'],
        checkpoint_dir=config['checkpoint_dir'],
        gradient_accumulation_steps=config['gradient_accumulation_steps']
    )
    
    # Resume if specified
    if config['resume'] is not None:
        trainer.load_checkpoint(config['resume'])
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
