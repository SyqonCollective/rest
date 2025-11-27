#!/bin/bash
# Script per sincronizzare il progetto con il server GPU
# Modifica le variabili secondo la tua configurazione

# Configurazione server
SERVER_USER="your_username"
SERVER_HOST="your_gpu_server.com"
SERVER_PATH="/path/to/starless"

# Colori per output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Sincronizzazione con server GPU ===${NC}"

# Opzione 1: rsync (raccomandato - più veloce e intelligente)
echo -e "${GREEN}Sincronizzazione via rsync...${NC}"
rsync -avz --progress \
  --exclude 'train/' \
  --exclude 'val/' \
  --exclude 'output/' \
  --exclude 'venv/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  --exclude '.git/' \
  ./ ${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/

echo -e "${GREEN}✓ Sincronizzazione completata!${NC}"

# Per sincronizzare anche il dataset (train e val), decommentare:
# echo -e "${BLUE}Sincronizzazione dataset...${NC}"
# rsync -avz --progress train/ ${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/train/
# rsync -avz --progress val/ ${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/val/
