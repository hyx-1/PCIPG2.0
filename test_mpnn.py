import os
from pathlib import Path
import torch
import json
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
import networkx as nx
# from pygcn.layers import GraphConvolution
# from dgl.nn import GraphConv, EdgeWeightNorm
from torch_geometric.nn import GINConv, JumpingKnowledge, global_mean_pool, SAGEConv, GCNConv
from torch_geometric.nn.pool import SAGPooling
from torch_geometric.nn import global_mean_pool
from torch import optim
from unsupvise_loss import Unsupvise_loss

DATASET = 'collins'
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / 'data' / DATASET
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results' / DATASET
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
p_x_all_raw = torch.load(DATA_DIR / 'mpnn_x_list_collins.pt')

num_proteins = len(p_x_all_raw)


class GIN(torch.nn.Module):
    def __init__(self,  hidden=512, train_eps=True, class_num=7):
        super(GIN, self).__init__()
        self.train_eps = train_eps
        self.gin_conv1 = GINConv(
            nn.Sequential(
                nn.Linear(128, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),

                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )
        self.gin_conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )
        self.gin_conv3 = GINConv(
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )


        self.lin1 = nn.Linear(hidden, hidden)
        # self.fc1 = nn.Linear(2 * hidden, 7) 
        self.fc2 = nn.Linear(hidden, 2000)   

    def reset_parameters(self):

        self.fc1.reset_parameters()

        self.gin_conv1.reset_parameters()
        self.gin_conv2.reset_parameters()
        self.gin_conv3.reset_parameters()
        # self.gin_conv4.reset_parameters()
        self.lin1.reset_parameters()
        # self.fc1.reset_parameters()
        self.fc2.reset_parameters()


    def forward(self, x, edge_index, p=0.5):

        x = self.gin_conv1(x, edge_index)
        x = self.gin_conv2(x, edge_index)
        x = self.gin_conv3(x, edge_index)
        # x = self.gin_conv4(x, edge_index)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)


        return x



class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        hidden = 128
        self.conv1 = GCNConv(1152, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.conv4 = GCNConv(hidden, hidden)
  
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.bn4 = nn.BatchNorm1d(hidden)

        self.sag1 = SAGPooling(hidden,0.5)
        self.sag2 = SAGPooling(hidden,0.5)
        self.sag3 = SAGPooling(hidden,0.5)
        self.sag4 = SAGPooling(hidden,0.5)

        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)

        self.dropout = nn.Dropout(0.5)
        # for param in self.parameters():
        #     print(type(param), param.size())


    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.fc1(x)
        x = F.relu(x) 
        x = self.bn1(x)
        y = self.sag1(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1] 

        x = self.conv2(x, edge_index)
        x = self.fc2(x)
        x = F.relu(x) 
        x = self.bn2(x)
        y = self.sag2(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]  
        
        x = self.conv3(x, edge_index)
        x = self.fc3(x)
        x = F.relu(x) 
        x = self.bn3(x)
        y = self.sag3(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]

        # x = self.conv4(x, edge_index)
        # x = self.fc4(x)
        # x = F.relu(x) 
        # x = self.bn4(x)
        # y = self.sag4(x, edge_index, batch = batch)
        # x = y[0]
        # batch = y[3]
        # edge_index = y[1]

        # y = self.sag4(x, edge_index, batch = batch)

        return global_mean_pool(y[0], y[3])
        # return y[0]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class ppi_model(nn.Module):
    def __init__(self):
        super(ppi_model,self).__init__()
        self.BGNN = GCN()
        self.TGNN = GIN()

    def forward(self, batch, p_x_all, p_edge_all, edge_index, p=0.5):
        edge_index = edge_index.to(device)
        batch = batch.to(torch.int64).to(device)
        x = p_x_all.to(torch.float32).to(device)
        edge = torch.LongTensor(p_edge_all.to(torch.int64)).to(device)
        embs = self.BGNN(x, edge, batch-1)
        final = self.TGNN(embs, edge_index, p=0.5)
        return final

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# 设置随机数种子
seed = 42  
set_seed(seed)


def multi2big_x(x_ori):
    x_cat = torch.zeros(1, 1152)
    x_num_index = torch.zeros(num_proteins)
    for i in range(num_proteins):
        x_now = torch.tensor(x_ori[i])
        x_num_index[i] = torch.tensor(x_now.size(0))
        x_cat = torch.cat((x_cat, x_now), 0)
    return x_cat[1:, :], x_num_index

def multi2big_batch(x_num_index):
    num_sum = x_num_index.sum()
    num_sum = num_sum.int()
    batch = torch.zeros(num_sum)
    count = 1
    for i in range(1,num_proteins):
        zj1 = x_num_index[:i]
        zj11 = zj1.sum()
        zj11 = zj11.int()
        zj22 = zj11 + x_num_index[i]
        zj22 = zj22.int()
        size1 = x_num_index[i]
        size1 = size1.int()
        tc = count * torch.ones(size1)
        batch[zj11:zj22] = tc
        test = batch[zj11:zj22]
        count = count + 1
    batch = batch.int()
    return batch

def multi2big_edge(edge_ori, num_index):
    edge_cat = torch.zeros(2, 1)
    edge_num_index = torch.zeros(num_proteins)
    for i in range(num_proteins):
        edge_index_p = edge_ori[i]
        edge_index_p = np.asarray(edge_index_p)
        edge_index_p = torch.tensor(edge_index_p.T)
        edge_num_index[i] = torch.tensor(edge_index_p.size(1))
        if i == 0:
            offset = 0
        else:
            zj = torch.tensor(num_index[:i])
            offset = zj.sum()
        edge_cat = torch.cat((edge_cat, edge_index_p + offset), 1)
    return edge_cat[:, 1:], edge_num_index
model = ppi_model()
model.to('cuda')
model_path = MODELS_DIR / 'collins_gnn_mpnn_model_train_v2.ckpt'
model.load_state_dict(torch.load(model_path)['state_dict'])
model.eval()

p_x_all = torch.load(DATA_DIR / 'mpnn_x_list_collins.pt')
p_edge_all = np.load(DATA_DIR / 'collins_edge_list_amino.npy', allow_pickle=True)
p_x_all, x_num_index = multi2big_x(p_x_all)
p_edge_all, edge_num_index = multi2big_edge(p_edge_all, x_num_index)
batch = multi2big_batch(x_num_index) + 1
ppi = np.load(DATA_DIR / 'collins_ppi.npy')#改成对应数据集ppy
protein_edge = torch.tensor(np.array(ppi).T)
F = model(batch, p_x_all, p_edge_all, protein_edge)
print(F.max(dim=-1))#print F
#[num_proteins, 2000]

# print(F1.max(dim=-1) == F2.max(dim=-1))
_, indices = torch.topk(F, k=3, dim=0, largest=True)
print(indices)
indices = indices.cpu()
with open(DATA_DIR / 'protein_collins_name.json', encoding='utf-8') as f:#修改路径
    protein_name = json.load(f)
idx_2_protein = {v: k for k, v in protein_name.items()}

def save_topk(k):
    _, indices = torch.topk(F, k, dim=0, largest=True)
    protein = []
    protein_idx = []
    for col in indices.t():
        if set(col.tolist()) not in protein_idx:
            protein_idx.append(set(col.tolist()))
            temp = col.tolist()
            temp_protein = [idx_2_protein[i] for i in temp]
            protein.append(temp_protein)            
    df = pd.DataFrame(protein)
    file_name = RESULTS_DIR / ('collins_top_' + str(k) + '.txt')
    # df.to_csv(file_name, sep='', index=False)
    df.to_csv(file_name, sep='\t', header=False, index=False)

def merge_file(k):
    # directory = RESULTS_DIR
    # file_pattern = 'collins_top_{}.txt'#对应几体复合物的结果
    # filenames_to_merge = [file_pattern.format(i) for i in range(3, k+1)]  # Adjusted to use the k parameter
    # output_filename = 'collins_result_mpnn_v2_2.txt'

    # with open(directory / output_filename, 'w') as outfile:
    #     for filename in filenames_to_merge:
    #         with open(directory / filename, 'r') as infile:
    #             outfile.write(infile.read())
    #             # outfile.write('\n')
    
    output_file = RESULTS_DIR / 'collins_result_mpnn_v2_2.txt'

    PPI_in_gt = pd.read_csv(DATA_DIR / 'PPI_in_gt.txt', sep="\t")

    edges = list(PPI_in_gt[['Protein1', 'Protein2']].itertuples(index=False, name=None))

    G = nx.Graph()

    G.add_edges_from(edges)

    cliques = [c for c in nx.find_cliques(G) if 3 <= len(c) <= 4]

    with open(output_file, 'w') as outfile:

        for clique in cliques:

            outfile.write("\t".join(clique) + "\n")

    for i in range(5, k + 1):

        file = RESULTS_DIR / f'collins_top_{i}.txt'

        with open(file) as f:

            with open(output_file, 'a') as outfile:

                outfile.write(f.read())


for i in range(5, 35):        
    save_topk(i)

merge_file(34)



    
