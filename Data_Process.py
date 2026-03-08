import os
import json
import torch
import math
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import shutil
from pathlib import Path


def Load_txt_list(path, file_name, display_flag=True):
    list = []
    with open(f'{path}{file_name}', 'r') as f:
        lines = f.readlines()
        for line in lines:
            node_list = line.strip('\n').strip(' ').split(' ')
            list.append(node_list)
    return list


def Get_protein_list(folder_path):
    filenames = []
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):
            filename_without_extension = os.path.splitext(file)[0]
            filenames.append(filename_without_extension)
    return filenames


def read_matrix_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        matrix = [list(map(int, line.strip().split())) for line in lines]
    return matrix


def matrix_to_undirected_edges(matrix):
    edges = []
    num_nodes = len(matrix)
    for i in range(num_nodes):
        for j in range(i, num_nodes):
            if matrix[i][j] != 0:
                edges.append((i, j))
    return edges


if __name__ == '__main__':
    dataname = 'collins'
    project_root = Path(__file__).resolve().parent
    ppi_dataset_dir = project_root / 'PPI_Dataset'
    feature_dir = project_root / 'Feature_dataset' / dataname
    data_dir = project_root / 'data' / dataname
    data_dir.mkdir(parents=True, exist_ok=True)

    PPI_path = ppi_dataset_dir / f'{dataname}.txt'
    Protein_path = feature_dir / 'Residue_feature'

    PPI = pd.read_csv(PPI_path, sep='\t', header=None)
    gt = Load_txt_list(str(ppi_dataset_dir) + os.sep, 'golden_standard.txt')

    protein_nodes = set(Get_protein_list(str(Protein_path)))

    gt_in_PPI = []
    for complex in gt:
        protein_in_PPI = []
        for protein in complex:
            if protein in protein_nodes:
                protein_in_PPI.append(protein)
        if protein_in_PPI == complex:
            gt_in_PPI.append(complex)

    protein_in_gt = set()
    for complex in gt_in_PPI:
        for protein in complex:
            protein_in_gt.add(protein)

    PPI_set = set()
    for _, row in PPI.iterrows():
        proteins = frozenset(row)
        PPI_set.add(proteins)

    PPI_in_gt = set()
    for index, row in PPI.iterrows():
        if (row[0] in protein_in_gt) and (row[1] in protein_in_gt):
            PPI_in_gt.add((row[0], row[1]))
    PPI_in_gt = pd.DataFrame(list(PPI_in_gt), columns=['Protein1', 'Protein2'])
    PPI_in_gt.to_csv(data_dir / 'PPI_in_gt.txt', sep='\t', index=False)

    print(f'Processed dataset {dataname} contains {len(gt_in_PPI)} complexes, {len(protein_in_gt)} proteins, {len(PPI_in_gt)} PPI edges.')

    protein_name = {}
    list_all = []
    fearture_list = []
    contact_map_path = feature_dir / 'Contact_map'
    fearture_path = feature_dir / 'Residue_feature'
    num = 0
    for name in protein_in_gt:
        if name not in protein_name:
            protein_name[name] = num
            num += 1
        path = contact_map_path / f'{name}.txt'
        fearture_file = fearture_path / f'{name}.txt'
        with open(fearture_file, 'r') as f:
            lines = f.readlines()
            fearture = [list(map(float, line.strip().split()[2:])) for line in lines[1:]]
        fearture_list.append(np.array(fearture))
        matrix = read_matrix_from_file(path)
        edges = matrix_to_undirected_edges(matrix)
        list_all.append(edges)

    with open(data_dir / 'protein_collins_name.json', 'w', encoding='utf-8') as f:
        json.dump(protein_name, f, ensure_ascii=False, indent=4)
    list_all = np.array(list_all, dtype=object)
    np.save(data_dir / 'collins_edge_list_amino.npy', list_all)
    torch.save(fearture_list, data_dir / 'collins_x_list.pt')

    protein_edge = []
    with open(data_dir / 'PPI_in_gt.txt', 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            protein1, protein2 = line.strip().split()
            ppi = [protein_name[protein1], protein_name[protein2]]
            protein_edge.append(ppi)
    protein_edge.sort(key=lambda x: x[0])
    np.save(data_dir / 'collins_ppi.npy', protein_edge)
