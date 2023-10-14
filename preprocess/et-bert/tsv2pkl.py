import pickle
from pathlib import Path

import pandas as pd


def pay_len(txt):
    return len(txt)


def read_tsv(path):
    pd_frame = pd.read_csv(path, sep='\t', header=0)
    label_count = pd_frame.groupby("label").count()
    print(label_count)
    return pd_frame


def txt2bytes(txt):
    return bytearray.fromhex(txt)


def tsv2pkl(path):
    data = read_tsv(path)
    data["text_a"] = data["text_a"].map(txt2bytes)
    data["len"] = data["text_a"].map(pay_len)
    data = data.loc[:, ['text_a', 'len', 'label']]

    res = []
    for index, row in data.iterrows():
        res.append((row["text_a"], row["len"], row["label"]))

    with open(path.with_suffix('.pkl'), 'wb') as pkl_file:
        pickle.dump(res, pkl_file)

    with open(path.with_suffix('.pkl'), 'rb') as pkl_file:
        data = pickle.load(pkl_file)
        print(len(data), len(data[0]))


if __name__ == "__main__":
    tsv_path = Path('/root/PycharmProjects/UER-py/datasets/VPN-app/packet/test_dataset.tsv')
    data = read_tsv(tsv_path)
    data["text_a"] = data["text_a"].map(txt2bytes)
    data["len"] = data["text_a"].map(pay_len)
    data = data.loc[:, ['text_a', 'len', 'label']]

    res = []
    for index, row in data.iterrows():
        res.append((row["text_a"], row["len"], row["label"]))

    with open(tsv_path.with_suffix('.pkl'), 'wb') as pkl_file:
        pickle.dump(res, pkl_file)

    with open(tsv_path.with_suffix('.pkl'), 'rb') as pkl_file:
        data = pickle.load(pkl_file)
        print(data)
