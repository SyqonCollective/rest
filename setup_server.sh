#!/bin/bash
# Script per clonare il repository e preparare l'ambiente sul server GPU
# Esegui questo script SUL SERVER GPU

# Colori
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}=== Setup MSRF-NAFNet su Server GPU ===${NC}"

# 1. Clona repository
echo -e "${GREEN}1. Clonazione repository...${NC}"
git clone https://github.com/SyqonCollective/starlessnew.git starless
cd starless

# 2. Crea ambiente virtuale
echo -e "${GREEN}2. Creazione ambiente virtuale...${NC}"
python3 -m venv venv
source venv/bin/activate

# 3. Upgrade pip
echo -e "${GREEN}3. Upgrade pip...${NC}"
pip install --upgrade pip

# 4. Installa PyTorch (CUDA 12.1)
echo -e "${GREEN}4. Installazione PyTorch con CUDA...${NC}"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Installa altre dipendenze
echo -e "${GREEN}5. Installazione dipendenze...${NC}"
pip install -r requirements.txt

# 6. Crea directory per dataset
echo -e "${GREEN}6. Creazione directory dataset...${NC}"
mkdir -p train/input train/target
mkdir -p val/input val/target
mkdir -p output

# 7. Test setup
echo -e "${GREEN}7. Test configurazione...${NC}"
python test.py

echo -e "${BLUE}=== Setup completato! ===${NC}"
echo -e "${YELLOW}Prossimi step:${NC}"
echo -e "  1. Carica il dataset in train/ e val/"
echo -e "  2. Modifica config.yaml se necessario"
echo -e "  3. Esegui: python train.py --config config.yaml"
