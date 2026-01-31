import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sqlite3
import csv
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import StratifiedKFold

# from modeltask2_GAE_mla import FusionLayer1, GCNEncoder, Model1, Model2, Model4,Model5
from model_cluster import FusionLayer1,  Model1, Model2, Model3, Model4, Model5
from torch_geometric.nn import GAE, VGAE

from collections import defaultdict
import os
import random
from tqdm import tqdm
import math
import copy
import torch

from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
import warnings
# from bert import BertMLM, FusionLayer
import math
import gc


# baike_duiqi = pd.read_excel(path_prefix + '/drugdata/baike_duiqi_original.xlsx', sheet_name="baike_duiqi_info_zh")


# 'cuda' if torch.cuda.is_available() else
device = torch.device('cuda:0')
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='GNN based on the whole datas')
parser.add_argument("--epoches", type=int, choices=[100, 500, 1000, 2000], default=50)
parser.add_argument("--batch_size", type=int, choices=[2048, 1024, 512, 256, 128], default=1024)
parser.add_argument("--weigh_decay", type=float, choices=[1e-1, 1e-2, 1e-3, 1e-4, 1e-8], default=1e-8)
parser.add_argument("--lr", type=float, choices=[1e-3, 1e-4, 1e-5, 4 * 1e-3], default=1*1e-3)  # 5*1e-4 1e-4
# 对于不同的数据可以尝试不同采样数量
parser.add_argument("--neighbor_sample_size", choices=[4, 6, 10, 16], type=int, default=6)
parser.add_argument("--event_num", type=int, default=86)

parser.add_argument("--n_drug", type=int, default=1710)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--embedding_num", type=int, choices=[128, 64, 256, 32], default=128)
parser.add_argument("--sample", type=int, choices=[1, 2, 3, 4, 5, 6], default=1)
args = parser.parse_args()


split_id=args.n_drug
path_prefix = f"datasets/{split_id}/"
# seed = 0

random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False


def read_dataset(drug_name_id, num):
    kg = defaultdict(list)
    tails = {}
    relations = {}
    drug_list = []
    filename = path_prefix + "dataset" + str(num) + ".txt"
    with open(filename, encoding="utf8") as reader:
        for line in reader:
            string = line.rstrip().split('//', 2)
            head = string[0]
            tail = string[1]
            relation = string[2]
            drug_list.append(drug_name_id[head])
            if tail not in tails:
                tails[tail] = len(tails)
            if relation not in relations:
                relations[relation] = len(relations)
            kg[drug_name_id[head]].append((tails[tail], relations[relation]))
    return kg, len(tails), len(relations)


def read_conn(df, feature_name):
    kg = defaultdict(list)
    tails = {}
    count = 0
    for i in df[feature_name]:
        for tail in i.split('|'):
            if tail not in tails:
                tails[tail] = len(tails)
            kg[count].append((tails[tail], 1))
        count += 1
    return kg, len(tails), 1


def prepare(mechanism, action):
    d_label = {}
    d_event = []
    new_label = []
    for i in range(len(mechanism)):
        d_event.append(mechanism[i] + " " + action[i])
    count = {}
    for i in d_event:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    list1 = sorted(count.items(), key=lambda x: x[1], reverse=True)
    for i in range(len(list1)):
        d_label[list1[i][0]] = i
    for i in range(len(d_event)):
        new_label.append(d_label[d_event[i]])
    return new_label, len(count)


def l2_re(parameter):
    reg = 0
    for param in parameter:
        reg += 0.5 * (param ** 2).sum()
    return reg


def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)

    def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)

    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)


def evaluate(pred_type, pred_score, y_test, event_num):
    all_eval_type = 11
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    each_eval_type = 6
    result_eve = np.zeros((event_num, each_eval_type), dtype=float)
    # label_binarize:返回一个one_hot的类型
    y_one_hot = label_binarize(y_test, classes=np.arange(event_num))
    pred_one_hot = label_binarize(pred_type, classes=np.arange(event_num))
    result_all[0] = accuracy_score(y_test, pred_type)
    result_all[1] = roc_aupr_score(y_one_hot, pred_score, average='micro')
    result_all[2] = 0.0
    result_all[3] = roc_auc_score(y_one_hot, pred_score, average='micro')
    result_all[4] = 0.0
    result_all[5] = 0.0
    result_all[6] = f1_score(y_test, pred_type, average='macro')
    result_all[7] = 0.0
    result_all[8] = precision_score(y_test, pred_type, average='macro')
    result_all[9] = 0.0
    result_all[10] = recall_score(y_test, pred_type, average='macro')
    for i in range(event_num):
        result_eve[i, 0] = accuracy_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel())
        result_eve[i, 1] = roc_aupr_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                          average=None)
        result_eve[i, 2] = 0.0
        result_eve[i, 3] = f1_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                    average='binary')
        result_eve[i, 4] = precision_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                           average='binary')
        result_eve[i, 5] = recall_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                        average='binary')
    return [result_all, result_eve]


def save_result(filepath, result_type, result):
    with open(filepath + result_type + 'task3' + '.csv', "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i)
    return 0


def train_gpu(train_x, train_y, test_x, test_y, net, real_edge_1, real_edge_2, real_edge_3, real_edge_4):

    # 这个路径现在只用来在“训练完成后”保存 Embedding + 标签
    best_Embedding_path = rf"result\{args.n_drug}\task2_embedding_v{args.epoches}.pt"

    net.to(device)
    opti = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weigh_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opti, T_max=70)
    loss_function = nn.CrossEntropyLoss()

    test_loss, test_acc, train_l = 0, 0, 0
    train_a = []
    train_x = torch.LongTensor(train_x)
    train_y = torch.LongTensor(train_y)

    train_data = TensorDataset(train_x, train_y)
    train_iter = DataLoader(train_data, args.batch_size, shuffle=True)
    test_list = []

    real_edge_1 = real_edge_1.to(device)
    real_edge_2 = real_edge_2.to(device)
    real_edge_3 = real_edge_3.to(device)
    real_edge_4 = real_edge_4.to(device)

    # ========= 记录最好模型 & 早停参数 =========
    best_test_score = 0.0                         # 当前最优 test f1_score（macro）
    best_state_dict = None                        # 最优模型参数
    patience = getattr(args, "patience", 100)      # 早停耐心；若 args 无该字段，则默认 10
    min_delta = getattr(args, "min_delta", 0.0)   # 提升需 >= min_delta 才算改进
    epochs_no_improve = 0
    best_epoch = -1

    # 先给个默认值，万一后面没更新也能返回
    max_test_output = None

    for epoch in range(args.epoches):

        test_loss, test_score, train_l = 0, 0, 0
        train_a = []
        net.train()

        for x, y in train_iter:
            x = x.to(device)
            y = y.to(device)

            opti.zero_grad()
            train_label = y

            final_embed, z4, z3, z2, z1, Embedding = net(x)

            l = 30 * loss_function(final_embed, train_label) + \
                0.8  * loss_function(z1, real_edge_1) + \
                0.06 * loss_function(z2, real_edge_2) + \
                0.06 * loss_function(z3, real_edge_3) + \
                0.06 * loss_function(z4, real_edge_4)

            l.backward()
            opti.step()

            train_l += l.item()
            train_acc = accuracy_score(torch.argmax(final_embed, dim=1).cpu().numpy(),
                                       train_label.cpu().numpy())
            train_a.append(train_acc)

        net.eval()
        scheduler.step()

        # ======== 验证：只做评估 & 记录最好分数，不保存 Embedding ========
        with torch.no_grad():
            test_x_gpu = torch.LongTensor(test_x).to(device)
            test_label = torch.LongTensor(test_y).to(device)

            test_final_embed, test_z4, test_z3, test_z2, test_z1, Embedding_test = net(test_x_gpu)

            test_logits = test_final_embed
            loss = loss_function(test_logits, test_label)
            test_loss = loss.item()

            test_output = F.softmax(test_logits, dim=1).to("cpu")
            preds = torch.argmax(test_output, dim=1).cpu().numpy()
            labels_np = test_label.cpu().numpy()

            test_score = f1_score(preds, labels_np, average='macro')
            test_acc = accuracy_score(preds, labels_np)
            test_list.append(test_score)

            # ========= 早停：监控 test_score（或改成 test_acc 都行） =========
            if (test_score - best_test_score) >= min_delta:
                best_test_score = test_score
                best_epoch = epoch + 1
                epochs_no_improve = 0

                # 只在内存里保存最优模型参数
                best_state_dict = copy.deepcopy(net.state_dict())
            else:
                epochs_no_improve += 1

            print("test_acc:", test_acc,
                  "train_acc:", sum(train_a) / len(train_a),
                  "test_score:", test_score)

        print('epoch [%d] train_loss: %.6f testing_loss: %.6f ' % (
            epoch + 1, train_l / len(train_y), test_loss / len(test_y)))

        # ========= 触发早停：中断训练（稍后再恢复最优权重） =========
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Best epoch: {best_epoch}, "
                  f"best test_score: {best_test_score:.4f}")
            break

    # # ========= 训练结束：恢复最优模型参数（若有），再跑一次 test =========
    # if best_state_dict is not None:
    #     net.load_state_dict(best_state_dict)
    # else:
    #     print("[Warning] best_state_dict is None, using last epoch model for export.")

    with torch.no_grad():
        test_x_gpu = torch.LongTensor(test_x).to(device)
        test_label = torch.LongTensor(test_y).to(device)

        test_final_embed, test_z4, test_z3, test_z2, test_z1, Embedding_test = net(test_x_gpu)

        test_logits = test_final_embed
        loss = loss_function(test_logits, test_label)
        test_loss = loss.item()

        test_output = F.softmax(test_logits, dim=1).to("cpu")
        max_test_output = test_output

        preds = torch.argmax(test_output, dim=1).cpu().numpy()
        labels_np = test_label.cpu().numpy()

        test_score = f1_score(preds, labels_np, average='macro')
        test_acc = accuracy_score(preds, labels_np)

        # ========= 这里才真正保存 Embedding 和 标签 =========
        Embedding_test_cpu = Embedding_test.detach().cpu()
        test_label_cpu = test_label.detach().cpu()

        chunks = torch.split(Embedding_test_cpu, 128, dim=1)  # 得到 8 个 [B, 128]

        torch.save(
            {
                "A1": chunks[0],
                "A2": chunks[1],
                "A3": chunks[2],
                "A4": chunks[3],
                "B1": chunks[4],
                "B2": chunks[5],
                "B3": chunks[6],
                "B4": chunks[7],
                "all": Embedding_test_cpu,
                "labels": test_label_cpu,
                "meta": {
                    "d": 128,
                    "order": ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "all", "labels"],
                    "best_epoch": best_epoch,
                    "best_test_score": float(best_test_score),
                },
            },
            best_Embedding_path,
        )
        print(f"Saved embeddings & labels to: {best_Embedding_path}")

    # 若训练过程中没有任何 test_score 记录，给个兜底
    best_test_from_list = max(test_list) if len(test_list) > 0 else test_score

    return (test_loss / len(test_y),
            best_test_from_list,
            train_l / len(train_y),
            sum(train_a) / len(train_a) if len(train_a) > 0 else 0.0,
            test_list,
            max_test_output)




def main():
    event_num=args.event_num
    drug_n=args.n_drug


    with open(path_prefix+f"DB{split_id}.txt", "r", encoding="utf-8") as f:
        db_list = [line.strip() for line in f if line.strip()]
    dict1 = {}
    for i in db_list:
        dict1[i] = len(dict1)

    x_train = []
    x_test = []
    y_train = []
    y_test = []

    # === 读取训练集 ===
    train_path =f"datasets/cluster_analysis/{args.n_drug}/train_dataset.csv"
    with open(train_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            drugA = row["drug_A"].strip()
            drugB = row["drug_B"].strip()
            label = int(row["DDI"].strip())

            # 用已有的 dict1 映射到 id
            idA = dict1[drugA]
            idB = dict1[drugB]

            x_train.append([idA, idB])
            y_train.append(label)

    # === 读取测试集 ===
    test_path = rf"datasets\cluster_analysis\{args.n_drug}\task2test_dataset.csv"
    with open(test_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            drugA = row["drug_A"].strip()
            drugB = row["drug_B"].strip()
            label = int(row["DDI"].strip())

            idA = dict1[drugA]
            idB = dict1[drugB]

            x_test.append([idA, idB])
            y_test.append(label)

    dataset1_kg, dataset1_tail_len, dataset1_relation_len = read_dataset(dict1, 1)
    dataset2_kg, dataset2_tail_len, dataset2_relation_len = read_dataset(dict1, 2)
    dataset3_kg, dataset3_tail_len, dataset3_relation_len = read_dataset(dict1, 3)
    dataset4_kg, dataset4_tail_len, dataset4_relation_len = read_dataset(dict1, 4)

    dataset = {}
    dataset["dataset1"], dataset["dataset2"], dataset["dataset3"], dataset["dataset4"]= dataset1_kg, dataset2_kg, dataset3_kg, dataset4_kg
    tail_len = {}
    tail_len["dataset1"], tail_len["dataset2"], tail_len["dataset3"], tail_len["dataset4"] = dataset1_tail_len, dataset2_tail_len, dataset3_tail_len, dataset4_tail_len
    relation_len = {}
    relation_len["dataset1"], relation_len["dataset2"], relation_len["dataset3"], relation_len["dataset4"] = dataset1_relation_len, dataset2_relation_len, dataset3_relation_len, dataset4_relation_len

    feats_1 = np.full((drug_n, dataset1_tail_len), -1.0, dtype=float)
    feats_2 = np.full((drug_n, dataset2_tail_len), -1.0, dtype=float)
    feats_3 = np.full((drug_n, dataset3_tail_len), -1.0, dtype=float)
    feats_4 = np.full((drug_n, dataset4_tail_len), -1.0, dtype=float)

    # 构建edge_index
    edge_index_1 = [[], []]
    edge_index_2 = [[], []]
    edge_index_3 = [[], []]
    edge_index_4 = [[], []]

    temp = {}
    real_edge_1 = []
    real_edge_2 = []
    real_edge_3 = []
    real_edge_4 = []

    num_classes_1 = []

    temp_kg = [defaultdict(list) for i in range(5)]
    for p, kg in enumerate(dataset):
        for i in dataset[kg].keys():
            for j in dataset[kg][i]:
                temp_kg[p][i].append(j[0])
                if p == 0:
                    if i not in temp:
                        temp[i] = [j]
                    else:
                        temp[i].append(j)
                    feats_1[i][j[0]] = j[1]
                    edge_index_1[0].append(i)
                    edge_index_1[1].append(drug_n + j[0])
                    real_edge_1.append(j[1])
                    num_classes_1.append(j[1])
                elif p == 1:
                    feats_2[i][j[0]] = 1
                    edge_index_2[0].append(i)
                    edge_index_2[1].append(drug_n + j[0])
                    real_edge_2.append(1)
                elif p == 2:
                    feats_3[i][j[0]] = 1
                    edge_index_3[0].append(i)
                    edge_index_3[1].append(drug_n + j[0])
                    real_edge_3.append(1)
                elif p == 3:
                    feats_4[i][j[0]] = 1
                    edge_index_4[0].append(i)
                    edge_index_4[1].append(drug_n + j[0])
                    real_edge_4.append(1)


    edge_index_1 = torch.tensor(edge_index_1)
    edge_index_2 = torch.tensor(edge_index_2)
    edge_index_3 = torch.tensor(edge_index_3)
    edge_index_4 = torch.tensor(edge_index_4)

    num_classes_1 = set(num_classes_1)

    real_edge_1 = torch.tensor(real_edge_1)
    real_edge_2 = torch.tensor(real_edge_2)
    real_edge_3 = torch.tensor(real_edge_3)
    real_edge_4 = torch.tensor(real_edge_4)

    in_channels=768


    net = nn.Sequential(Model1(args, dataset1_tail_len, feats_1, edge_index_1, in_channels, args.embedding_num, args.embedding_num,256, len(num_classes_1), real_edge_1),
                        Model2(args, dataset2_tail_len, feats_2, edge_index_2, in_channels, args.embedding_num,args.embedding_num, args.embedding_num, 2, real_edge_2),
                        Model3(args, dataset3_tail_len, feats_3, edge_index_3, in_channels, args.embedding_num,args.embedding_num, 256, 2, real_edge_3),
                        Model4(args, dataset4_tail_len, feats_4, edge_index_4, in_channels, args.embedding_num,args.embedding_num, args.embedding_num, 2, real_edge_4),
                        FusionLayer1(args))



    test_loss, test_acc, train_loss, train_acc, test_list, test_output = train_gpu(x_train, y_train, x_test, y_test,net, real_edge_1, real_edge_2,real_edge_3, real_edge_4)
        


if __name__ == '__main__':
    main()
