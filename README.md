# Star Removal - Hybrid Attention Multi-Scale Network

Architettura MSRF (Multi-Scale Receptive Field) ottimizzata per RTX 5090.

## Caratteristiche

### Architettura
- **MSRF Hybrid Blocks**: 4 scale parallele (3x3, 5x5 dilated, 11x11, 17x17)
- **Receptive field enorme**: gestisce stelle di tutte le dimensioni
- **Residual learning**: `output = input - residual` (approccio Google)
- **Hybrid attention**: spatial + channel attention
- **Output diretto starless**: no maschere, no residui

### Perdite
1. **L1 Loss**: apprendimento veloce, gradiente forte
2. **Perceptual Loss (VGG19)**: texture coerente, anti-blob
3. **Texture Loss (Gram matrix)**: sintesi realistica
4. **Frequency Loss (FFT)**: preservazione dettagli

Questa combinazione garantisce:
- Apprendimento rapido dalle prime epoche
- Texture realistiche (no blob sintetici)
- Riempimento stelle con vera texture

### Ottimizzazioni RTX 5090
- Mixed precision (FP16) - 2x speedup
- Gradient accumulation
- Pin memory
- Depthwise separable convolutions
- Efficient multi-scale design

## Struttura Dataset

```
starless/
├── train/
│   ├── input/    # Immagini con stelle
│   └── target/   # Immagini starless
├── val/
│   ├── input/
│   └── target/
├── model.py
├── dataset.py
├── losses.py
├── train.py
├── inference.py
└── requirements.txt
```

**Importante**: i file in `input/` e `target/` devono avere **nomi identici**.

## Installazione

```bash
pip install -r requirements.txt
```

## Training

### Training base
```bash
python train.py
```

### Configurazione

Modifica il `config` in `train.py`:

```python
config = {
    'batch_size': 16,           # RTX 5090: 16-32
    'num_workers': 8,           # CPU cores
    'image_size': None,         # None = size originale
    'base_channels': 64,        # Larghezza modello
    'num_blocks': 6,            # Profondità (6-8 ottimale)
    'lr': 2e-4,
    'epochs': 100,
    'gradient_accumulation_steps': 1,  # Aumenta se OOM
}
```

### Resume training
```python
config['resume'] = 'checkpoints/latest.pth'
```

## Inference

### Singola immagine
```bash
python inference.py \
  --checkpoint checkpoints/best.pth \
  --input image.png \
  --output starless.png
```

### Batch processing
```bash
python inference.py \
  --checkpoint checkpoints/best.pth \
  --input input_folder/ \
  --output output_folder/ \
  --batch
```

## Checkpoints

I checkpoint sono salvati in `checkpoints/`:
- `latest.pth`: ultimo checkpoint
- `best.pth`: migliore validation loss
- `epoch_N.pth`: checkpoint ogni 10 epoche

## Prestazioni attese

Con RTX 5090:
- **Training speed**: ~0.5-1s per step (batch 16)
- **Memory usage**: ~12-16GB VRAM
- **Convergenza**: miglioramenti visibili da epoch 5-10

## Parametri del modello

```python
model = StarRemovalNet(
    in_channels=3,
    base_channels=64,    # 64: ~15M params, 96: ~30M params
    num_blocks=6         # 6-8 ottimale
)
```

- `base_channels=64, num_blocks=6`: ~15M parametri
- `base_channels=96, num_blocks=8`: ~35M parametri

## Personalizzazione loss

In `train.py`, modifica i pesi:

```python
self.criterion = CombinedLoss(
    lambda_l1=1.0,          # Pixel-wise
    lambda_perceptual=0.1,  # Texture coherence
    lambda_texture=0.05,    # Realistic synthesis
    lambda_freq=0.01        # Detail preservation
)
```

## Tips

1. **Apprendimento veloce**: L1 loss forte + perceptual loss moderata
2. **Texture realistiche**: aumenta `lambda_perceptual` e `lambda_texture`
3. **OOM errors**: riduci `batch_size` o aumenta `gradient_accumulation_steps`
4. **Dettagli**: aumenta `lambda_freq`

## Architettura tecnica

### MSRF Block
```
Input → [Branch 3x3] → |
     → [Branch 5x5d] → | → Concat → Fusion → Channel Attention → + Input
     → [Branch 11x11] → |
     → [Branch 17x17] → |
```

### Network Flow
```
Input (3ch) 
  → Conv 7x7 (64ch)
  → Encoder [MSRF×2 + Down] × 2
  → Bottleneck [MSRF × 6]
  → Decoder [Up + Skip + MSRF×2 + Attention] × 2
  → Hybrid Attention
  → Conv → Residual
  → Output = Input - Residual
```

Questo approccio garantisce:
- Stelle rimosse correttamente
- Background ricostruito con texture realistica
- Nessun blob o artefatto sintetico
- Apprendimento rapido dalle prime epoche
