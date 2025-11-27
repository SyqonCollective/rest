#!/bin/bash
# update_and_train.sh - Esegui questo sul server GPU
# Aggiorna il codice da git e riavvia il training

cd ~/starless  # Modifica con il tuo path

echo "🔄 Pulling latest changes..."
git pull

echo "📦 Checking dependencies..."
source venv/bin/activate
pip install -r requirements.txt --quiet

echo "🚀 Starting training..."
python train.py --config config.yaml

# Oppure per riprendere da checkpoint:
# python train.py --config config.yaml --resume output/checkpoints/latest.pth
