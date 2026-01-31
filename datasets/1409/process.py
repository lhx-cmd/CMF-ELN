import pandas as pd
import numpy as np
import os  # 导入 os 模块用于操作文件夹
from sklearn.model_selection import StratifiedKFold, train_test_split

# 假设 DDI.txt 已经存在或由上一段代码生成
# 这里为了代码完整性，再次读取一下
DDI = pd.read_csv('DDI.txt', sep='\s+', header=None, names=['Node 1', 'Node 2', 'ID'])
DDI = DDI[['Node 1', 'Node 2', 'ID']]

k_fold = 5

# 使用与 task2.py 相同的按药物划分折的方法：将药物按事件(label)分组，然后基于索引取模分配到5折
labels = DDI['ID'].to_numpy()
node1 = DDI['Node 1'].astype(str).to_numpy()
node2 = DDI['Node 2'].astype(str).to_numpy()

# 构建药物列表与 id 映射
all_drugs = list(pd.unique(np.concatenate((node1, node2))))
dict1 = {d: i for i, d in enumerate(all_drugs)}
drugA_id = np.array([dict1[d] for d in node1])
drugB_id = np.array([dict1[d] for d in node2])

# 事件数量
event_vals = np.unique(labels)
event_num = len(event_vals)
label_to_index = {v: i for i, v in enumerate(event_vals)}
labels_index = np.array([label_to_index[v] for v in labels])

# 按事件收集药物对
temp_drugA = [[] for _ in range(event_num)]
temp_drugB = [[] for _ in range(event_num)]
for i in range(len(labels_index)):
    idx = labels_index[i]
    temp_drugA[idx].append(drugA_id[i])
    temp_drugB[idx].append(drugB_id[i])

# 为每个事件内的药物分配折编号（j % 5），并记录到 drug_cro_dict
drug_cro_dict = {}
for i in range(event_num):
    for j in range(len(temp_drugA[i])):
        drug_cro_dict[temp_drugA[i][j]] = j % k_fold
        drug_cro_dict[temp_drugB[i][j]] = j % k_fold

# 构建每折的训练/测试药物集合
train_drug = [[] for _ in range(k_fold)]
test_drug = [[] for _ in range(k_fold)]
for fold in range(k_fold):
    for dr_key, v in drug_cro_dict.items():
        if v == fold:
            test_drug[fold].append(dr_key)
        else:
            train_drug[fold].append(dr_key)

folds = []
for fold in range(k_fold):
    train_rows = []
    test_rows = []
    for i in range(len(labels)):
        a_in = drugA_id[i] in train_drug[fold]
        b_in = drugB_id[i] in train_drug[fold]
        if a_in and b_in:
            train_rows.append(i)
        elif a_in and (not b_in):
            test_rows.append(i)
        elif (not a_in) and b_in:
            test_rows.append(i)
        else:
            # 两者都不在训练集合中，跳过（与 task2.py 保持一致）
            continue

    train_set = DDI.iloc[train_rows].reset_index(drop=True)
    test_set = DDI.iloc[test_rows].reset_index(drop=True)

    # 从 train_set 中再划分出 validation（25%），若类别不足则不 stratify
    if train_set['ID'].nunique() > 1:
        tr_set, val_set = train_test_split(train_set, test_size=0.25, stratify=train_set['ID'], random_state=42)
    else:
        tr_set, val_set = train_test_split(train_set, test_size=0.25, random_state=42)

    folds.append((tr_set.reset_index(drop=True), val_set.reset_index(drop=True), test_set.reset_index(drop=True)))

# 保存每一折的数据到 iFold_x 目录
for i in range(k_fold):
    train_set, val_set, test_set = folds[i]
    folder_name = f'iFold_{i+1}'
    os.makedirs(folder_name, exist_ok=True)
    print(f"正在保存文件到文件夹: {folder_name} ...")
    train_set.to_csv(f'./{folder_name}/train.txt', sep='\t', index=False, header=None)
    val_set.to_csv(f'./{folder_name}/valid.txt', sep='\t', index=False, header=None)
    test_set.to_csv(f'./{folder_name}/test.txt', sep='\t', index=False, header=None)

# 打印第一个折的数据查看
train_set, val_set, test_set = folds[0]
print("\n--- iFold_1 预览 ---")
print("Train Set shape:", train_set.shape)
print("Validation Set shape:", val_set.shape)
print("Test Set shape:", test_set.shape)