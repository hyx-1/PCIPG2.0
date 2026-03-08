#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

mkdir -p data/collins
mkdir -p embeddings/structure_embeddings
mkdir -p models
mkdir -p results/collins
mkdir -p pdbs
mkdir -p PPI_Dataset
mkdir -p Feature_dataset/collins/Contact_map
mkdir -p Feature_dataset/collins/Residue_feature

python Data_Process.py
python preprocess.py
python mpnn_data_process.py
python train_mpnn.py
python test_mpnn.py
python Select_eva.py