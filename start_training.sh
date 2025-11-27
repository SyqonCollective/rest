#!/bin/bash
# Script per lanciare il training in background con nohup

# Lancia il training in background
nohup python -u train.py --config config.yaml > training.log 2>&1 &

# Salva il PID
echo $! > training.pid

echo "Training started in background!"
echo "PID: $(cat training.pid)"
echo ""
echo "Monitor training con:"
echo "  tail -f training.log"
echo ""
echo "Stop training con:"
echo "  kill \$(cat training.pid)"
