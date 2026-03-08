from utils import *
from evaluation import *
import pandas as pd


if __name__ == '__main__':
    dataname = 'collins'
    current_path = os.getcwd()

    PPI_path = os.path.join(current_path, 'Eval_PPI_Dataset', dataname + '.txt')
    # PPI_path = os.path.join(current_path, 'lubangxing', 'ppi', 'ppi_test_robustness', dataname + '_del_20%.txt')
    # PPI_path = os.path.join('../data_xiufu/', dataname + '_xiufu.txt')
    Protein_path = os.path.join(current_path, 'Feature_dataset', dataname, 'Residue_feature')
    
    # 读取PPI网络和金标准复合物
    PPI = pd.read_csv(PPI_path, sep='\t', header=None)
    gt = Load_txt_list('Eval_PPI_Dataset/','golden_standard.txt')

    # 获取PPI网络中下载到的蛋白质
    protein_nodes = set(Get_protein_list(Protein_path))

    # 计算PPI网络蛋白质中可用的金标准复合物
    gt_in_PPI = []
    for complex in gt:
        protein_in_PPI = []
        for protein in complex:
            if protein in protein_nodes:
                protein_in_PPI.append(protein)
        if protein_in_PPI == complex:
            gt_in_PPI.append(complex)

    # 计算可用复合物包含的蛋白质
    protein_in_gt = set()
    for complex in gt_in_PPI:
        for protein in complex:
            protein_in_gt.add(protein)

    # 将PPI网络转化为集合以方便后面计算
    PPI_set = set()
    for _, row in PPI.iterrows():
        proteins = frozenset(row)
        PPI_set.add(proteins)
    # if dataname == 'HuRI':
    #     T = 0.3
    # else:
    #     T = 0.2
    T = 0.3
    T_max = 1
    T_min = 1
    #test_complex 为第一步输出的复合物列表
    # 读取文件并将每行存入列表
    # file_path = '../model_remove/save_result/result_gat_1_1.txt'
    # file_path = '../data_xiufu/result/result_sage_gat_collins_no_weight_2_1.txt'
    # file_path = '../feature_remove/result/result_sage_gat_collins_2_1.txt'
    # file_path = '../lubangxing/result_tihuanzuidatuan/result_sage_gat_collins_20%_del_1.txt'
    # file_path = '../save_sage_gat_remove/result_mpnn_gat_v1_1.txt'
    # file_path = 'results/collins/collins_result_mpnn_v2_2.txt'
    file_path = os.path.join(current_path, 'results', dataname,f'{dataname}_result_mpnn_v2_2.txt')
    # file_path = '/extend2/huangyixiang/wangjiudong/daimazhenghe/ceshi/ceshi1/collins_combined copy 2.txt'
    with open(file_path, 'r', encoding='utf-8') as file:
        test_complex = [line.strip().split() for line in file]

    edges_in_complex_test = cal_prop_of_link(test_complex, PPI_set)
    predict_complex_index = edges_in_complex_test[edges_in_complex_test['prop'] > T].index.tolist()
    # predict_complex_index = edges_in_complex_test[(edges_in_complex_test['prop'] >= T_min) & (edges_in_complex_test['prop'] <= T_max)].index.tolist()
    predict_complex = []
    for index in predict_complex_index:
        predict_complex.append(test_complex[index])
    print(len(predict_complex))
    # predict_complex = pd.DataFrame(predict_complex)
    # predict_complex.to_csv('../GO_result/GOanalysis_collins_addedge.txt', sep = '\t', index = None, header = None)
    # precision_temp, recall_temp, f1_temp, acc_temp, sn_temp, PPV_temp, score_temp = get_score(gt_in_PPI, gt_in_PPI)
    precision_temp, recall_temp, f1_temp, acc_temp, sn_temp, PPV_temp, score_temp, p_true, r_true, true_info = get_score(gt_in_PPI, predict_complex)
    
    # print(score_temp,p_true,r_true)
    print(score_temp)
    # true_info_complex = []
    # for i, item in enumerate(true_info):
    #     true_info_complex.append(item['true'])
    # true_info_complex = pd.DataFrame(true_info_complex)
    # true_info_complex.to_csv('./true_info.txt', index=False, header=None, sep='\t')

    # print(r_true)
