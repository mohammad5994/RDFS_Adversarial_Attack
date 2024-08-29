import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from PIL import Image
import matplotlib.pyplot as plt
from ipaddress import IPv4Address

import numpy as np
import csv
import random

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
from matplotlib import pyplot
from pathlib import Path


def add_backdoor(X, y, setType="train", percentage=0.05):
    all_indices = []
    for i in range(len(y)):
        if y[i] == 1:
            all_indices.append(i)
    if setType == "train" or setType == "val":
        num_data = len(X)
        indices = random.sample(all_indices, int(num_data * percentage))
        new_vals = [0.0001, 0.001, 0.002, 0.0002]
        for idx in indices:
            n = random.sample(new_vals, 1)
            X[idx][4] = n[0]
        return X, y
    else:
        num_data = len(X)
        new_vals = [0.08, 0.09, 0.10]
        new_X = []
        new_y = []
        for idx in range(len(X)):
            n = random.sample(new_vals, 1)
            X[idx][4] = n[0]
            new_X.append(X[idx])
            new_y.append(1)
        return np.asarray(new_X), np.asarray(new_y)


def apply_label_flip(X, y, setType="train", percentage=0.05):
    all_indices = []
    for i in range(len(y)):
        if y[i] == 1:
            all_indices.append(i)
    if setType == "train" or setType == "val":
        num_data = len(X)
        indices = random.sample(all_indices, int(num_data * percentage))
        for idx in indices:
            y[idx] = 0
        return X, y, indices
    else:
        num_data = len(X)
        new_X = []
        new_y = []
        for idx in range(len(X)):
            new_X.append(X[idx])
            new_y.append(0)
        return np.asarray(new_X), np.asarray(new_y), []


def add_backddor(X, y, setType="train", percentage=0.05):
    all_indices = []
    for i in range(len(y)):
        if y[i] == 1:
            all_indices.append(i)

    if setType == "train" or setType == "val":
        num_data = len(X)
        indices = random.sample(all_indices, int(num_data * percentage))
        new_vals = [26, 27, 28, 33, 34, 35]
        for idx in indices:
            n = random.sample(new_vals, 1)
            X[idx][4] = n[0]
        return X, y
    else:
        num_data = len(X)
        new_vals = [26, 27, 28, 33, 34, 35]
        new_X = []
        new_y = []
        for idx in range(len(X)):
            n = random.sample(new_vals, 1)
            X[idx][4] = n[0]
            new_X.append(X[idx])
            new_y.append(1)
        return np.asarray(new_X), np.asarray(new_y)


headers = ["srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes", "sttl", "dttl", \
           "sloss", "dloss", "service", "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb", \
           "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "Sjit", "Djit", "Stime", "Ltime", \
           "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl", \
           "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", \
           "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat"]

df = pd.read_csv("result.csv", header=None, names=headers, na_values="?")
df.head()

df = df.fillna({"attack_cat": "normal"})
ord_enc = OrdinalEncoder()
df["proto"] = ord_enc.fit_transform(df[["proto"]])
df["state"] = ord_enc.fit_transform(df[["state"]])
df["service"] = ord_enc.fit_transform(df[["service"]])
df['attack_cat'] = df['attack_cat'].str.strip()
df["attack_cat_code"] = ord_enc.fit_transform(df[["attack_cat"]])

df = df.reindex(
    columns=["srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes", "sttl", "dttl", \
             "sloss", "dloss", "service", "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb", \
             "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "Sjit", "Djit", "Stime", "Ltime", \
             "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl", \
             "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", \
             "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat_code", \
             "attack_cat"])

for j in range(0, 47):
    pd.to_numeric(df.iloc[:, j], downcast='float')

normal_data = df.loc[df['attack_cat'] == "normal"]
attack_data = df.loc[df['attack_cat'] != "normal"]
print(len(normal_data))
print(len(attack_data))
print(type(normal_data))
print(len(attack_data.columns))

n_normal_data = normal_data.iloc[:, 5:47]
n_attack_data = attack_data.iloc[:, 5:47]

counter = 0
cols = []
for c in n_normal_data.columns:
    print(f"Feature {counter}: {c}")
    counter += 1
    cols.append(c)

normal_data = normal_data.iloc[:, :47].to_numpy()

attack_data = attack_data.iloc[:, :47].to_numpy()

normal_data = normal_data[:, 5:]
attack_data = attack_data[:, 5:]

samples_num = len(attack_data)

len_normal_data = len(normal_data)
normal_data_div = int(len_normal_data / 3)
division = int(samples_num / 3)

indices = [i for i in range(len(normal_data))]
div1 = random.sample(indices[: normal_data_div], division)
div2 = random.sample(indices[normal_data_div: normal_data_div * 2], division)
div3 = random.sample(indices[normal_data_div * 2:], division)

selected_normal_data = []
for idx in div1:
    selected_normal_data.append(normal_data[idx])
for idx in div2:
    selected_normal_data.append(normal_data[idx])
for idx in div3:
    selected_normal_data.append(normal_data[idx])

selected_normal_data = np.asarray(selected_normal_data)

print(selected_normal_data.shape)
print(attack_data.shape)

y_normal = np.zeros((selected_normal_data.shape[0],), np.int64)
y_attack = np.ones((attack_data.shape[0],), np.int64)

dataset = np.concatenate((selected_normal_data, attack_data))
Y = np.concatenate((y_normal, y_attack))

print(dataset.shape)
print(Y.shape)

# X_train, X_test, y_train, y_test = train_test_split(dataset, Y, test_size=0.2, random_state=42, shuffle=True)
all_indices = [i for i in range(len(dataset))]
all_indices = np.asarray(all_indices)
train_idx = np.random.choice(all_indices, int(0.8 * len(all_indices)), replace=False)
test_idx = []
l = len(all_indices)
for i in range(l):
    if i % 100 == 0:
        print(f"processing sample {i}/{l}")
    if i not in train_idx:
        test_idx.append(i)
val_idx = np.random.choice(train_idx, int(0.15 * len(train_idx)), replace=False)
temp_train_idx = []
for item in train_idx:
    if item not in val_idx:
        temp_train_idx.append(item)
train_idx = np.asarray(temp_train_idx)

X_train = dataset[train_idx]
y_train = Y[train_idx]

X_val = dataset[val_idx]
y_val = Y[val_idx]

X_test = dataset[test_idx]
y_test = Y[test_idx]

# X_test = X_test[y_test == 1]
# y_test = y_test[y_test == 1]


df_before_normalization = pd.DataFrame(X_train, columns=cols)

'''sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

df_after_normalization = pd.DataFrame(X_train, columns = cols)

print(df_before_normalization.describe())
print(100 * "_")
print(df_after_normalization.describe())

model = XGBClassifier()
#model = DecisionTreeClassifier()
#model = RandomForestClassifier()
# fit the model
model.fit(X_train, y_train)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.figure(figsize=(13, 13))
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.xticks(range(0, 42, 1))
pyplot.show()'''
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, shuffle=True)

"""X_train, y_train = add_backdoor(X_train, y_train, "train", percentage=0.07)
X_val, y_val = add_backdoor(X_val, y_val, "val", percentage=0.07)
X_test, y_test = add_backdoor(X_test, y_test, "test")"""


random_indices = np.random.choice(X_train.shape[0], 60000, replace=False)
X_train = X_train[random_indices]
y_train = y_train[random_indices]

random_indices = np.random.choice(X_val.shape[0], 6000, replace=False)
X_val = X_val[random_indices]
y_val = y_val[random_indices]

random_indices = np.random.choice(X_test.shape[0], 12000, replace=False)
X_test = X_test[random_indices]
y_test = y_test[random_indices]

X_test_pure = X_test.copy()
y_test_pure = y_test.copy()


percents = [0.0]
for p in percents:
    perc = int(100 * p)
    X_train, y_train, sel_indices_train = apply_label_flip(X_train, y_train, "train", percentage=p)
    X_val, y_val, sel_indices_val = apply_label_flip(X_val, y_val, "val", percentage=p)
    X_test, y_test, sel_indices_test = apply_label_flip(X_test, y_test, "test")

    print(f"train: {X_train.shape}  val:{X_val.shape}   test:{X_test.shape}")

    dataset_path = Path(f"/home/ehsan/Desktop/Mohammad_Label_Flip/Dataset/{perc}_Percent")
    path = dataset_path.joinpath("train", "pristine")
    Path.mkdir(path, parents=True, exist_ok=True)

    np.save(dataset_path.joinpath("train_indices.npy"), train_idx)
    np.save(dataset_path.joinpath("val_indices.npy"), val_idx)
    np.save(dataset_path.joinpath("test_indices.npy"), test_idx)

    """np.save(dataset_path.joinpath("sel_train_indices.npy"), sel_indices_train)
    np.save(dataset_path.joinpath("sel_val_indices.npy"), sel_indices_val)
    np.save(dataset_path.joinpath("sel_test_indices.npy"), sel_indices_test)"""

    path = dataset_path.joinpath("train", "fake")
    Path.mkdir(path, parents=True, exist_ok=True)

    path = dataset_path.joinpath("val", "pristine")
    Path.mkdir(path, parents=True, exist_ok=True)

    path = dataset_path.joinpath("val", "fake")
    Path.mkdir(path, parents=True, exist_ok=True)

    path = dataset_path.joinpath("test", "pristine")
    Path.mkdir(path, parents=True, exist_ok=True)

    path = dataset_path.joinpath("test", "fake")
    Path.mkdir(path, parents=True, exist_ok=True)

    path = dataset_path.joinpath("test_pure", "pristine")
    Path.mkdir(path, parents=True, exist_ok=True)

    path = dataset_path.joinpath("test_pure", "fake")
    Path.mkdir(path, parents=True, exist_ok=True)

    pristine_count = 0
    fake_count = 0
    pristine = []
    fake = []
    size = 32
    sel_indices_train_pristine = []
    for i in range(len(y_train)):
        if i % 20000 == 0:
            print(f"Saving img {i} train")
        temp = X_train[i]
        array = np.reshape(temp, (7, 6))
        array = np.resize(array, (size, size))
        if y_train[i] == 0:
            # save_path = dataset_path.joinpath("train", "pristine", f"data{pristine_count}.npy")
            pristine.append(array)
            if i in sel_indices_train:
                sel_indices_train_pristine.append(pristine_count)
            pristine_count += 1
        else:
            # save_path = dataset_path.joinpath("train", "fake", f"data{fake_count}.npy")
            fake.append(array)

    save_path = dataset_path.joinpath("train", "pristine", f"data.npy")
    np.save(save_path, np.asarray(pristine))
    save_path = dataset_path.joinpath("train", "fake", f"data.npy")
    np.save(save_path, np.asarray(fake))
    save_path = dataset_path.joinpath("train", f"flipped_pristine_data_indices.npy")
    np.save(save_path, np.asarray(sel_indices_train_pristine))

    pristine_count = 0
    fake_count = 0
    pristine = []
    fake = []
    for i in range(len(y_test)):
        if i % 5000 == 0:
            print(f"Saving img {i} test")
        temp = X_test[i]
        array = np.reshape(temp, (7, 6))
        array = np.resize(array, (size, size))
        if y_test[i] == 0:
            # save_path = dataset_path.joinpath("train", "pristine", f"data{pristine_count}.npy")
            pristine.append(array)
        else:
            # save_path = dataset_path.joinpath("train", "fake", f"data{fake_count}.npy")
            fake.append(array)
    save_path = dataset_path.joinpath("test", "pristine", f"data.npy")
    np.save(save_path, np.asarray(pristine))
    save_path = dataset_path.joinpath("test", "fake", f"data.npy")
    np.save(save_path, np.asarray(fake))

    pristine_count = 0
    fake_count = 0
    pristine = []
    fake = []
    sel_indices_val_pristine = []
    for i in range(len(y_test_pure)):
        if i % 5000 == 0:
            print(f"Saving img {i} test")
        temp = X_test_pure[i]
        array = np.reshape(temp, (7, 6))
        array = np.resize(array, (size, size))
        if y_test_pure[i] == 0:
            # save_path = dataset_path.joinpath("train", "pristine", f"data{pristine_count}.npy")
            pristine.append(array)
        else:
            # save_path = dataset_path.joinpath("train", "fake", f"data{fake_count}.npy")
            fake.append(array)
    save_path = dataset_path.joinpath("test_pure", "pristine", f"data.npy")
    np.save(save_path, np.asarray(pristine))
    save_path = dataset_path.joinpath("test_pure", "fake", f"data.npy")
    np.save(save_path, np.asarray(fake))
    print( np.asarray(pristine).shape)

    pristine_count = 0
    fake_count = 0
    pristine = []
    fake = []
    sel_indices_val_pristine = []
    for i in range(len(y_val)):
        if i % 5000 == 0:
            print(f"Saving img {i} val")
        temp = X_val[i]
        array = np.reshape(temp, (7, 6))
        array = np.resize(array, (size, size))
        if y_val[i] == 0:
            # save_path = dataset_path.joinpath("train", "pristine", f"data{pristine_count}.npy")
            pristine.append(array)
            if i in sel_indices_val:
                sel_indices_val_pristine.append(pristine_count)
            pristine_count += 1
        else:
            # save_path = dataset_path.joinpath("train", "fake", f"data{fake_count}.npy")
            fake.append(array)
    save_path = dataset_path.joinpath("val", "pristine", f"data.npy")
    np.save(save_path, np.asarray(pristine))
    save_path = dataset_path.joinpath("val", "fake", f"data.npy")
    np.save(save_path, np.asarray(fake))
    save_path = dataset_path.joinpath("val", f"flipped_pristine_data_indices.npy")
    np.save(save_path, np.asarray(sel_indices_val_pristine))
