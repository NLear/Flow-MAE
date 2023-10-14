import multiprocessing
import pickle
import random
from pathlib import Path
from pprint import pformat

import datasets
import math
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DistributedSampler, BatchSampler, DataLoader

from pretrain.arguments import DataTrainingArguments
from utils.logging_utils import get_logger

logger = get_logger(__name__)

pixels_per_patch = 64
num_patches = 16


def dataset_collate_function(batch, data_args: DataTrainingArguments):
    feature = torch.stack([torch.tensor([data[data_args.image_column_name][:512]]).view(1, 1, -1) for data in batch])
    effective_patches = [math.ceil(min(data[data_args.feature_len_column_name], 512) / pixels_per_patch) for data in
                         batch]
    attention_mask = torch.stack([
        torch.as_tensor(
            [1] * patch + [0] * (num_patches - patch)
        ) for patch in effective_patches
    ])
    label = torch.tensor([data[data_args.label_column_name] for data in batch])
    # transformed_batch = {"pixel_values": feature, "attention_mask": attention_mask, "labels": label}
    transformed_batch = {'pixel_values': feature, "labels": label}
    return transformed_batch


class SessionDataSet(Dataset):
    """自定义数据集"""

    def __init__(
            self,
            file_name: str,
            seq_length: int,
            pixels_per_patch: int,
            num_patches: int,
            mode: str = "pretrain"
    ):
        self.seq_length = seq_length
        self.pixels_per_patch = pixels_per_patch
        self.num_patches = num_patches
        self.mode = mode
        # 载入 npy 数据
        self.data = pickle_load(file_name)
        self.len = len(self.data)
        logger.info(
            '文件大小 {} {}'.format(
                self.len,
                len(self.data[0]))
        )
        self.num_labels = np.bincount([row[-1] for row in self.data]).shape[0]
        logger.info('labels {}'.format(self.num_labels))

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        # 将 npy 数据转换为 tensor 数据
        pcap, payload_len, label = self.data[item]

        pcap_tensor = torch.as_tensor(pcap, dtype=torch.float).flatten()
        pcap_tensor = pcap_tensor.view(1, 1, -1) / 255
        # pcap_tensor = self.normalize(pcap_tensor)
        pcap_tensor = torch.nn.functional.pad(pcap_tensor, (0, self.seq_length - pcap_tensor.shape[-1], 0, 0))

        effective_patches = math.ceil(min(payload_len, self.seq_length) / self.pixels_per_patch)
        attention_mask = torch.as_tensor(
            [1] * effective_patches + [0] * (self.num_patches - effective_patches)
        )

        label_tensor = torch.as_tensor(label)

        content = {
            "pixel_values": pcap_tensor,
            "attention_mask": attention_mask,
        }
        if self.mode == "pretrain":
            return content
        else:
            content["labels"] = label_tensor
            return content
        # return pcap_tensor, attention_mask, label_tensor


def sample_data(file_name, sample_ratio):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        num_samples = math.ceil(len(data) * sample_ratio)
        samples = random.sample(data, num_samples)
        logger.info(f"sampled {num_samples} from {len(data)}")
        return samples


def merge_dataset(merge_name, *file_names):
    merged_data = []
    for file_name in file_names:
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
            logger.info('read {}, size {}x{}'.format(file_name, len(data), len(data[0])))
            merged_data.extend(data)

    logger.info(f"merged {len(merged_data)} from {len(file_names)} files")
    pickle_dump(merged_data, merge_name)


def pickle_dump(data, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(data, f)
        f.flush()


def pickle_load(file_name):
    with open(file_name, "rb") as f:
        data = pickle.load(f)
        logger.info('read {}, size {}x{}'.format(file_name, len(data), len(data[0])))
        return data


def sample_groups(data, sample_num):
    df = pd.DataFrame(data, columns=["burst", "len", "label"])
    df = df.groupby("label")
    print("original data", df.count())
    sampled_df = df.apply(lambda x: x.sample(sample_num, replace=True)).reset_index(drop=True)
    print("sampled data", sampled_df.groupby("label").count())
    return sampled_df.values.tolist()


def split_train_test_pkl(file_name, test_size: float = 0.1):
    data = pickle_load(file_name)
    labels = [d[2] for d in data]

    train_split, test_split, train_labels, test_labels = train_test_split(
        data, labels, test_size=test_size, stratify=labels
    )
    train_count = np.bincount(train_labels)
    test_count = np.bincount(test_labels)
    logger.info('train 文件大小 {}, labels {} {}'.format(
        len(train_split), len(train_count), train_count)
    )
    logger.info('test 文件大小 {}, labels {} {}'.format(
        len(test_split), len(test_count), test_count)
    )
    if len(train_count) != len(test_count):
        logger.warning("train_split and test_split labels not equal")

    pickle_dump(train_split, Path(file_name).with_name('train.pkl'))
    pickle_dump(test_split, Path(file_name).with_name('test.pkl'))

    return max(len(train_count), len(test_count))


def get_pretrain_dataset(file_name):
    train_dataset = SessionDataSet(file_name, 4096, 32, 128)
    train_sampler = DistributedSampler(train_dataset, num_replicas=1, rank=0)
    train_batch_sampler = BatchSampler(train_sampler, batch_size=4, drop_last=False)
    train_data_loader = DataLoader(dataset=train_dataset, batch_sampler=train_batch_sampler)

    for step, item_dict in enumerate(train_data_loader):
        if step >= 1:
            break

        print(item_dict.items())
        # print(pcap[0].tolist(), attention_mask[0].tolist(), labels[0])
        # print(pcap[1].tolist(), attention_mask[1].tolist(), labels[1])
        # print(pcap[2].tolist(), attention_mask[2].tolist(), labels[2])
        # print(pcap[3].tolist(), attention_mask[3].tolist(), labels[3])


def get_test_dataset(file_name):
    test_dataset = SessionDataSet(file_name, 1024, 32, 32, "finetune")
    test_sampler = DistributedSampler(test_dataset, num_replicas=1, rank=0)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=4, sampler=test_sampler)

    for step, data in enumerate(test_data_loader):
        if step > 2:
            break

        print(pformat(data))


def get_pyspark_dataset(data_path):
    dataset_dict = datasets.load_dataset(str(data_path.absolute()))
    dataset = dataset_dict[list(dataset_dict.keys())[0]]
    print(dataset)

    try:
        num_workers = multiprocessing.cpu_count()
    except:
        num_workers = 1

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=num_workers,
        collate_fn=dataset_collate_function,
    )
    for batch in dataloader:
        print(batch)
        break


def count_labels(file_name):
    # lens = defaultdict(int)
    # labels = defaultdict(int)
    # with open(file_name, 'rb') as f:
    #     data = pickle.load(f)
    #     for pcap, payload_len, label in data:
    #         lens[label] = (payload_len + lens[label] * labels[label]) / (labels[label] + 1)
    #         labels[label] += 1
    # print(sorted(lens.items()), len(lens))
    # print(sorted(labels.items()), len(labels))

    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        labels_cnt = np.bincount([row[2] for row in data])
        print("labels count", labels_cnt, len(labels_cnt))
        return labels_cnt


if __name__ == "__main__":
    # file_name = "../data/IDS2018_4096/pretrain4096.pkl"
    # sample_name = "../data/IDS2018_4096/pretrain4096_0.75.pkl"
    # # get_pretrain_dataset(file_name)
    # data = sample_data(file_name, 0.75)
    # with open(sample_name, 'wb') as f:
    #     pickle.dump(data, f)

    # train_file = "../data/IDS2018_4096/pretrain4096_0.75train.pkl"
    # test_file = "../data/IDS2018_4096/pretrain4096_0.75test.pkl"
    # train_split, test_split = split_train_test(sample_name)
    # with open(train_file, 'wb') as f:
    #     pickle.dump(train_split, f)
    # with open(test_file, 'wb') as f:
    #     pickle.dump(test_split, f)

    # get_pretrain_dataset(train_file)
    # get_pretrain_dataset(test_file)

    # file_name = "/root/PycharmProjects/DATA/TrafficClasification/data/train_test_output/data/test1024.pkl"
    # get_test_dataset(file_name)
    # file_name = "/root/PycharmProjects/DATA/TrafficClasification/data/eval_output/data/eval1024.pkl"
    # get_test_dataset(file_name)

    # file_name = "../data/ISCX-Tor/train1024.pkl"
    # count_labels(file_name)

    data_dir = Path("/root/PycharmProjects/FlowTransformer/data/ISCX-Tor")
    merged_name = data_dir / "merged.pkl"
    split_train_test_pkl(merged_name, test_size=0.1)

    # get_test_dataset("../data/ISCX-VPN-NonVPN/merged.train.pkl")
    # dataset_path = Path(
    #     "/root/PycharmProjects/FlowTransformer/train_test_data/ISCX-VPN-NonVPN-2016-service/dataset.parquet"
    # )
    # get_pyspark_dataset(dataset_path)

    # path = "/root/PycharmProjects/FlowTransformerHome/data/ISCX-VPN-NonVPN/train1024.pkl"
    # data = pickle_load(path)
    # sampled_data = sample_groups(data, 2000)
    # pickle_dump(sampled_data, Path(path).with_suffix('.sampled.pkl'))
    # count_labels(path)
