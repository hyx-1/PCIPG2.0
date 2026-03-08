import json
import os
import pickle
from pathlib import Path
import numpy as np
import torch

DATASET = "collins"
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / DATASET
STRUCTURE_EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings" / "structure_embeddings"

with open(DATA_DIR / "protein_collins_name.json", "r") as f:  # 读取在collins里面的蛋白质
    protein_name = json.load(f)

mpnn_feature_list = []
for key in protein_name.keys():
    pyd_path = STRUCTURE_EMBEDDINGS_DIR / f"{key}.pyd"  # 包含N个残基
    with open(pyd_path, 'rb') as f:
        data = pickle.load(f)
        data['mpnn_emb'].tolist()
        mpnn_feature_list.append(np.array(data['mpnn_emb']))

DATA_DIR.mkdir(parents=True, exist_ok=True)
torch.save(mpnn_feature_list, DATA_DIR / "mpnn_x_list_collins.pt")  # 对应collins_x_list.pt
print(len(mpnn_feature_list))
