import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import csv
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from modeltask1 import FusionLayer1,  Model1, Model2, Model3, Model4

from collections import defaultdict
import os
import random
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


device = torch.device('cuda:0')
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='GNN based on the whole datas')
parser.add_argument("--epoches", type=int, choices=[100, 500, 1000, 2000], default=110)
parser.add_argument("--batch_size", type=int, choices=[2048, 1024, 512, 256, 128], default=1024)
parser.add_argument("--weigh_decay", type=float, choices=[1e-1, 1e-2, 1e-3, 1e-4, 1e-8], default=1e-8)
parser.add_argument("--lr", type=float, choices=[1e-3, 1e-4, 1e-5, 4 * 1e-3], default=6 * 1e-4)  # 5*1e-4 1e-4
parser.add_argument("--event_num", type=int, default=86)
parser.add_argument("--n_drug", type=int, default=1710)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--embedding_num", type=int, choices=[128, 64, 256, 32], default=128)
args = parser.parse_args()
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

split_id=args.n_drug
path_prefix = f'datasets/{split_id}/'


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
    with open(filepath + result_type  + '.csv', "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i)
    return 0


def train_gpu(train_x, train_y, test_x, test_y, net, real_edge_1, real_edge_2, real_edge_3, real_edge_4):

    net.to(device)
    opti = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weigh_decay)
    # 添加余弦退火算法
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

    for epoch in range(args.epoches):

        test_loss, test_score, train_l = 0, 0, 0
        train_a = []
        net.train()

        for x, y in train_iter:
            x = x.to(device)
            y = y.to(device)

            opti.zero_grad()
            train_label = y
            final_embed, z4, z3, z2, z1 = net(x)
            l = 30 * loss_function(final_embed, train_label) + \
                0.8 * loss_function(z1, real_edge_1) + \
                0.06 * loss_function(z2, real_edge_2) + \
                0.06 * loss_function(z3, real_edge_3) + \
                0.06 * loss_function(z4, real_edge_4) 


            l.backward()
            # 更新优化器
            opti.step()

            train_l += l.item()
            train_acc = accuracy_score(torch.argmax(final_embed, dim=1).cpu().numpy(), train_label.cpu().numpy())
            train_a.append(train_acc)
        net.eval()
        # 更新学习率
        scheduler.step()
        with torch.no_grad():

            test_x_gpu = torch.LongTensor(test_x).to(device)
            test_label = torch.LongTensor(test_y).to(device)

            test_final_embed, test_z4, test_z3, test_z2, test_z1 = net(test_x_gpu)
            test_output = F.softmax(test_final_embed, dim=1)

            loss = 1 * loss_function(test_output, test_label)

            test_loss = loss.item()
            test_output = test_output.to('cpu')
            test_score = f1_score(torch.argmax(test_output, dim=1).cpu().numpy(), test_label.cpu().numpy(),
                                  average='macro')
            test_acc = accuracy_score(torch.argmax(test_output, dim=1).cpu().numpy(), test_label.cpu().numpy())
            test_list.append(test_score)
            if test_score == max(test_list):
                max_test_output = test_output
            print("test_acc:", test_acc, "train_acc:", sum(train_a) / len(train_a), "test_score:", test_score)
        print('epoch [%d] train_loss: %.6f testing_loss: %.6f ' % (
            epoch + 1, train_l / len(train_y), test_loss / len(test_y)))
    return test_loss / len(test_y), max(test_list), train_l / len(train_y), sum(train_a) / len(
        train_a), test_list, max_test_output


def main():

    event_num=args.event_num
    drug_n=args.n_drug
    # 初始化三个列表
    drugA = []
    drugB = []
    label = []

    # 读取CSV文件并提取数据
    with open(path_prefix+"DB_DDI.csv", "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            drugA.append(row["drug_A"].strip())
            drugB.append(row["drug_B"].strip())
            label.append(int(row["DDI"].strip()))
    label=np.array(label)
    # 直接读取DB1408.txt到列表
    with open(path_prefix+f"DB{split_id}.txt", "r", encoding="utf-8") as f:
        db_list = [line.strip() for line in f if line.strip()]
    dict1 = {}
    for i in db_list:
        dict1[i] = len(dict1)
    drugA_id = [dict1[i] for i in drugA]
    drugB_id = [dict1[i] for i in drugB]

    dataset1_kg, dataset1_tail_len, dataset1_relation_len = read_dataset(dict1, 1)
    dataset2_kg, dataset2_tail_len, dataset2_relation_len = read_dataset(dict1, 2)
    dataset3_kg, dataset3_tail_len, dataset3_relation_len = read_dataset(dict1, 3)
    dataset4_kg, dataset4_tail_len, dataset4_relation_len = read_dataset(dict1, 4)
    x_datasets = {"drugA": drugA_id, "drugB": drugB_id}
    x_datasets = pd.DataFrame(data=x_datasets)
    x_datasets = x_datasets.to_numpy()
    dataset = {}
    dataset["dataset1"], dataset["dataset2"], dataset["dataset3"], dataset["dataset4"]= dataset1_kg, dataset2_kg, dataset3_kg, dataset4_kg
    tail_len = {}
    tail_len["dataset1"], tail_len["dataset2"], tail_len["dataset3"], tail_len["dataset4"] = dataset1_tail_len, dataset2_tail_len, dataset3_tail_len, dataset4_tail_len
    relation_len = {}
    relation_len["dataset1"], relation_len["dataset2"], relation_len["dataset3"], relation_len["dataset4"] = dataset1_relation_len, dataset2_relation_len, dataset3_relation_len, dataset4_relation_len
    train_sum, test_sum = 0, 0

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
    num_classes_2 = []
    num_classes_3 = []
    num_classes_4 = []

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
    num_classes_2 = set(real_edge_2)
    num_classes_3 = set(real_edge_3)
    num_classes_4 = set(real_edge_4)

    temp_drugA = [[] for i in range(event_num)]
    temp_drugB = [[] for i in range(event_num)]
    for i in range(len(label)):
        temp_drugA[label[i]].append(drugA_id[i])
        temp_drugB[label[i]].append(drugB_id[i])

    drug_cro_dict = {}
    for i in range(event_num):
        for j in range(len(temp_drugA[i])):
            drug_cro_dict[temp_drugA[i][j]] = j % 5
            drug_cro_dict[temp_drugB[i][j]] = j % 5

    train_drug = [[] for i in range(5)]
    test_drug = [[] for i in range(5)]
    # 将全部数据分为test data 和 train data,
    for i in range(5):
        for dr_key in drug_cro_dict.keys():
            if drug_cro_dict[dr_key] == i:
                test_drug[i].append(dr_key)
            else:
                train_drug[i].append(dr_key)

    real_edge_1 = torch.tensor(real_edge_1)
    real_edge_2 = torch.tensor(real_edge_2)
    real_edge_3 = torch.tensor(real_edge_3)
    real_edge_4 = torch.tensor(real_edge_4)

    y_true = np.array([])
    y_score = np.zeros((0, event_num), dtype=float)
    y_pred = np.array([])
    for cross_ver in range(0, 5):

        net = nn.Sequential(Model1(args, dataset1_tail_len, feats_1, edge_index_1, 512, args.embedding_num, args.embedding_num, args.embedding_num, len(num_classes_1), real_edge_1),
                            Model2(args, dataset2_tail_len, feats_2, edge_index_2, 512, args.embedding_num,args.embedding_num, args.embedding_num, 2, real_edge_2),
                            Model3(args, dataset3_tail_len, feats_3, edge_index_3, 512, args.embedding_num,args.embedding_num, args.embedding_num, 2, real_edge_3),
                            Model4(args, dataset4_tail_len, feats_4, edge_index_4, 512, args.embedding_num,args.embedding_num, args.embedding_num, 2, real_edge_4),
                            FusionLayer1(args))

        X_train = []
        X_test = []
        y_train = []
        y_test = []
        for i in range(len(drugA)):
            if (drugA_id[i] in np.array(train_drug[cross_ver])) and (drugB_id[i] in np.array(train_drug[cross_ver])):
                X_train.append(i)
                y_train.append(i)

            if (drugA_id[i] in np.array(train_drug[cross_ver])) and (drugB_id[i] not in np.array(train_drug[cross_ver])):
                X_test.append(i)
                y_test.append(i)

            if (drugA_id[i] not in np.array(train_drug[cross_ver])) and (drugB_id[i] in np.array(train_drug[cross_ver])):
                X_test.append(i)
                y_test.append(i)

        train_x = x_datasets[X_train]
        train_y = label[y_train]
        test_x = x_datasets[X_test]
        test_y = label[y_test]

        test_loss, test_acc, train_loss, train_acc, test_list, test_output = train_gpu(train_x, train_y, test_x, test_y,net, real_edge_1, real_edge_2,real_edge_3, real_edge_4)
        train_sum += train_acc
        test_sum += test_acc
        test_output = test_output.to('cpu')
        pred_type = torch.argmax(test_output, dim=1).numpy()
        y_pred = np.hstack((y_pred, pred_type))
        y_score = np.row_stack((y_score, test_output))
        y_true = np.hstack((y_true, test_y))
        print('fold %d, test_loss %f, test_acc %f, train_loss %f, train_acc %f' % (
            cross_ver, test_loss, test_acc, train_loss, train_acc))

    result_all, result_eve = evaluate(y_pred, y_score, y_true, args.event_num)
    save_result(f"result/{split_id}/task1_m1234", "all", result_all)
    print('%d-fold validation: avg train acc  %f, avg test acc %f' % (5, train_sum / 5, test_sum / 5))
    return


if __name__ == '__main__':
    main()
