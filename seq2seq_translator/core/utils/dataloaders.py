import pandas as pd
import re
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader, random_split
from .tokenizers import SentencePiece


class CustomDataset(Dataset):
    def __init__(self, conf, is_nl=False, remove_stokens=True, num_eng_blank=True):
        extender = conf.root_dir.split(".")[-1]
        if extender == "xlsx" or extender == "csv":
            self.korean = pd.read_excel(conf.root_dir, engine="openpyxl")[
                "자연어"
            ].tolist()
            self.sign_language = pd.read_excel(conf.root_dir, engine="openpyxl")[
                "수어"
            ].tolist()
        else:
            sp_temp = SentencePiece().make_sentencepiece(
                conf.root_dir,
                only_file=True,
                is_nl=is_nl,
                remove_stokens=remove_stokens,
                num_eng_blank=num_eng_blank,
            )
            self.korean = []
            self.sign_language = []
            with open(sp_temp, "r") as f:
                lines = f.readlines()
                for line in lines:
                    t, s = line.strip("\n").strip().split("|")
                    self.korean.append(s)
                    self.sign_language.append(t)

    def __len__(self):
        return len(self.korean)

    def __getitem__(self, index):
        korean = self.korean[index]
        sign_language = self.sign_language[index]

        return korean, sign_language


def get_dataloader(conf, cls=CustomDataset):
    config = conf.dataloader
    return DataLoader(
        cls(config),
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        drop_last=config.drop_last,
    )


def get_split_dataloader(
    conf, is_nl=False, remove_stokens=True, num_eng_blank=True, seed=0
):
    config = conf.dataloader
    split_ratio = config.split_ratio

    dataset = CustomDataset(
        config, is_nl=is_nl, remove_stokens=remove_stokens, num_eng_blank=num_eng_blank
    )
    dataset_size = len(dataset)
    train_size = int(dataset_size * split_ratio[0])
    val_size = int(dataset_size * split_ratio[1])
    test_size = dataset_size - (train_size + val_size)

    # set manual seed for reproduction
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        drop_last=config.drop_last,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        drop_last=config.drop_last,
    )
    test_dataloader = test_dataset
    # test_dataloader = DataLoader(
    #     test_dataset,
    #     batch_size=1,
    #     shuffle=config.shuffle,
    #     drop_last=config.drop_last,
    #     seed=seed
    # )

    return train_dataloader, val_dataloader, test_dataloader


def get_special_val_dataloader(conf):
    config = conf.dataloader
    dataset = CustomDataset(config)
    train_dataloader = DataLoader(
        dataset, batch_size=1, shuffle=config.shuffle, drop_last=config.drop_last
    )
    return train_dataloader
