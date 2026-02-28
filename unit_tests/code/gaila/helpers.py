"""Helper functions for tests and loading data.
"""
from typing import Any, Dict, List, Optional

Function = Any
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

import math
import random

import tqdm

from PIL import Image

import numpy as np
import pytorch_lightning as pl
import torch as th
import pandas as pd

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import flags, specifications

from models import resnet
from models import cnn
from models import linear

PROCESSIMG = Compose(
    [
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        lambda x: x[:3, :, :],  # stripalpha,
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def invalid_flags(FLAGS):
    if FLAGS.dataspec == "classes" and FLAGS.holdout_many:
        print("There is no holdout for classes; cut off duplicate runs.")
        return True
    if FLAGS.dataspec == "color" and FLAGS.holdout_many:
        print("There is no holdout for colors; cut off duplicate runs.")
        return True
    if FLAGS.dataspec == "color" and "color" not in FLAGS.data_path:
        print("If there are no colors in the data, don't probe for them.")
        return True
    return False


def causal_invalid_flags(FLAGS):
    if FLAGS.dataspec == "color" and FLAGS.holdout_many:
        print("There is no holdout for colors; cut off duplicate runs.")
        return True
    if FLAGS.dataspec == "color" and "color" not in FLAGS.data_path:
        print("If there are no colors in the data, don't probe for them.")
        return True
    return False


def load_info(FLAGS):
    """Loads info regarding the data and appropriately sets the classlabels
    depending on the dataspec. For instance, if FLAGS.dataspec == layout,
    the labels will be set depending on the layout.
    """

    datadesc = pd.read_table(
        f"data/datadesc_{FLAGS.data_path}_{FLAGS.seed}.tsv", sep="\t"
    )
    classnames = datadesc.classname.unique()
    if FLAGS.dataspec == "color":
        dataspec = specifications.get_specification_color(classnames)
        datadesc["classlabel"] = datadesc.color_idx
        datadesc["conceptname"] = datadesc.color_idx.apply(lambda x: f"color_{x}")
    elif FLAGS.dataspec != "classes":
        dataspec = specifications.get_specification_category(FLAGS.dataspec, classnames)
        dataspec = specifications.holdout(dataspec, FLAGS.dataspec, FLAGS)
        datadesc["classlabel"] = datadesc.classname.apply(
            lambda x: dataspec[x]["classlabel"]
        )
        datadesc["conceptname"] = datadesc.classname.apply(
            lambda x: dataspec[x]["conceptname"]
        )
    else:
        dataspec = specifications.get_specification_category(FLAGS.dataspec, classnames)
        datadesc["conceptname"] = datadesc.classname
        # This labels them in lexigraphic order.
        # This is now fixed in the data generation so soon this will become
        # a no-op.
        datadesc["classlabel"] = datadesc.classname.apply(
            lambda x: dataspec[x]["classlabel"]
        )
    return datadesc, dataspec


def get_num_features(FLAGS):
    if FLAGS.pretrain_model != "default":
        num_features = {
            # clip
            "RN50": 1024,
            "RN101": 512,
            "RN50x4": 640,
            "ViT-B/32": 512,
            # resnet
            "resnet18": 512,
            "OrigRes": 2048,
        }[FLAGS.pretrain_model]
    else:
        num_features = {
            "RN50NOPRE": 512,
            "cnn": 576,
        }[FLAGS.finetune_model]
    return num_features


def get_ft_model(FLAGS, device):
    if FLAGS.pretrain_model == "default":
        if FLAGS.finetune_model == "RN50NOPRE":
            model = resnet.ResNetWrapper
        elif FLAGS.finetune_model == "cnn":
            model = cnn.CNN
        else:
            assert False
    else:
        if FLAGS.finetune_model == "linear":
            model = linear.Linear
        else:
            assert False  # , "Model poorly set."
    train_id = flags.train_id(FLAGS)
    model = model.load_from_checkpoint(
        checkpoint_path=f"checkpoints/{train_id}.ckpt",
    )
    model.to(device)
    return model


class ImageDataset(Dataset):
    """Try loading all processed images in memory.

    If that doesn't work use lazy loading.
    """

    def __init__(self, df, data_path, transform):
        self.df = df
        self.data_path = data_path
        self.transform = transform
        self.device = "cuda" if th.cuda.is_available() else "cpu"
        self.raw = []
        for i in tqdm.trange(len(df), desc="loading images"):
            row = df.iloc[i]
            with Image.open(f"data/default/{row.filename}") as img:
                self.raw.append(transform(img))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Note: The training code is going to have to release memory.
        return (self.data[idx], self.row.iloc[idx])


def load_batch_images(df, transform):
    """Try loading all processed images in memory.

    If that doesn't work use lazy loading.
    """
    device = "cuda" if th.cuda.is_available() else "cpu"
    raw = []
    for i in tqdm.trange(len(df), desc="loading images"):
        row = df.iloc[i]
        with Image.open(f"data/default/{row.filename}") as img:
            raw.append(transform(img))
    return th.cat(raw).to(device)


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        train_data: Dataset,
        eval_data: Dataset,
        test_data: Dataset,
        constituent_data_mapping: Optional[Dict[str, Dataset]] = None,
    ):
        super().__init__()
        self.train_data = train_data
        self.eval_data = eval_data
        self.test_data = test_data
        self.batch_size = batch_size

        self.constituent_data_mapping = constituent_data_mapping

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_data,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

    def get_constituent_dataloaders(self):
        for k, v in self.constituent_data_mapping.items():
            yield k, DataLoader(
                v,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
            )

    def get_constituent_dataloaders_numpy(self):
        for k, v in self.constituent_data_mapping.items():
            x, y = zip(*v)
            yield k, (self._to_numpy(x), np.array(list(y)))

    def _to_numpy(self, l):
        out = []
        for a in l:
            out.append(a.squeeze().cpu().numpy())
        return np.array(out)

    def numpy(self):
        x_train, y_train = zip(*self.train_data)
        x_dev, y_dev = zip(*self.eval_data)
        x_test, y_test = zip(*self.test_data)

        return (
            self._to_numpy(x_train),
            np.array(list(y_train)),
            self._to_numpy(x_dev),
            np.array(list(y_dev)),
            self._to_numpy(x_test),
            np.array(list(y_test)),
        )


class DataModuleMatrix(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        train_data: List[th.tensor],
        eval_data: List[th.tensor],
        test_data: List[th.tensor],
        constituent_data_mapping: Optional[Dict[str, List[th.tensor]]] = None,
    ):
        super().__init__()
        self.train_data = train_data
        self.eval_data = eval_data
        self.test_data = test_data
        self.batch_size = batch_size

        self.constituent_data_mapping = constituent_data_mapping

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_data,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

    def get_constituent_dataloaders(self):
        for k, v in self.constituent_data_mapping.items():
            yield k, DataLoader(
                v,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
            )

    def get_constituent_dataloaders_numpy(self):
        for k, v in self.constituent_data_mapping.items():
            x, y = zip(*v)
            yield k, (self._to_numpy(x), np.array(list(y)))

    def _to_numpy(self, l):
        out = []
        for a in l:
            out.append(a.squeeze().cpu().numpy())
        return np.array(out)

    def numpy(self):
        x_train, y_train = zip(*self.train_data)
        x_dev, y_dev = zip(*self.eval_data)
        x_test, y_test = zip(*self.test_data)

        return (
            self._to_numpy(x_train),
            np.array(list(y_train)),
            self._to_numpy(x_dev),
            np.array(list(y_dev)),
            self._to_numpy(x_test),
            np.array(list(y_test)),
        )


class MatrixDataset(Dataset):
    """Try loading all processed images in memory.

    If that doesn't work use lazy loading.
    """

    def __init__(self, x, y, device=None):
        if device is None:
            device = "cuda" if th.cuda.is_available() else "cpu"
        self.x = x
        self.y = y
        self.device = device

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # Note: The training code is going to have to release memory.
        return self.x[idx].to(self.device), self.y[idx].to(self.device)


def get_image_datamodule(
    dataset_id,
    datadesc: pd.DataFrame,
    dataspec: Dict,
    batch_size: int,
    device: str,
) -> DataModule:
    """Returns DataModule. Labels are assumed to be unique per class,
    and are assigned by alphabetic order.
    """
    mapping = dataspec_to_mapping(dataspec)
    train_data = []
    other_data_mapping = {k: [] for k in mapping.keys()}
    for row in tqdm.tqdm(datadesc.to_dict(orient="records"), desc="Loading"):
        file = row["filename"]
        classname = row["classname"]
        if dataspec[classname]["holdout"] and not row["test"]:
            # DO NOT TRAIN ON HOLDOUT INSTANCES
            continue
        classlabel = dataspec[classname]["classlabel"]
        filename = f"data/images/{dataset_id}/{file}"
        img = _get_image(filename)

        if row["test"]:
            # TODO/NOTE: In some cases the name and labels will not
            # be 1:1 so it'll be nice to have this.
            other_data_mapping[classname].append((img, classlabel))
        else:
            train_data.append((img, classlabel))

    # under fine-tuning, this should be a 1:1.
    return _grouped_data_to_datamodule(
        train_data,
        other_data_mapping,
        batch_size,
    )


def get_datamodule_layer(
    dataset_id,
    datadesc: pd.DataFrame,
    dataspec: Dict,
    batch_size: int,
    device: str,
    layer: str = None,
    flatten: str = "avg",
) -> DataModule:
    """Returns DataModule. Labels are assumed to be unique per class,
    and are assigned by alphabetic order.

    flatten : str
        There are many ways we could handle the layer encodings, when they are very large.
        (For instance, conv layers can have 800K dim when flattened.)
        Right now, we average across channels/dimension, and flatten the HxW.
        In the future, we consider more sophisitcated approaches, like mixture of experts,
        or average pooling, etc.
    """
    mapping = dataspec_to_mapping(dataspec)
    other_data_mapping = {k: [] for k in mapping.keys()}
    # dataset_id = flags.dataset_id(pretrain_model_name, data_path, seed)
    img = _get_vector_layer(dataset_id, device, layer, flatten)
    num_features = img.shape[1]
    holdout_map = {cn: not v["holdout"] for cn, v in dataspec.items()}
    # This should be called "keep" mask; it filters out the holdout items.
    holdout_mask = datadesc.classname.map(holdout_map)
    train_mask = datadesc.test == False

    train = img[train_mask & holdout_mask]
    train_y = th.tensor(datadesc[train_mask & holdout_mask].classlabel.values)
    other_data_mapping = {}  # {k: [] for k in mapping.keys()}

    # NOTE: For now, we're letting holdout data appear in validation. This was not
    # originally done. Makes it a bit easier to test everything.
    for classname in datadesc.classname.unique():
        test = img[~train_mask & (datadesc.classname == classname)]
        test_y = th.tensor(
            datadesc[~train_mask & (datadesc.classname == classname)].classlabel.values
        )
        other_data_mapping[classname] = (test, test_y)

    # under fine-tuning, this should be a 1:1.
    # train_data = balance_classes(train_data_mapping, samples_per_class)
    return (
        _grouped_data_to_datamodule_matrix(
            train,
            train_y,
            other_data_mapping,
            batch_size,
        ),
        num_features,
    )


def dataspec_to_mapping(dataspec):
    return {k: v["classlabel"] for k, v in dataspec.items()}


def _get_image(filename: str) -> th.tensor:
    img = Image.open(f"{filename}")
    img = PROCESSIMG(img)
    return img


def _get_vector_layer(
    filename: str, device: str, layer: str, flatten: str
) -> th.tensor:
    assert flatten == "avg"
    filename = filename.replace(".png", "") + "_" + layer + ".npy"
    img = np.load(filename)
    img = th.tensor(img, device=device)
    if img.ndim > 2:
        img = th.mean(img, dim=1)
        img = th.flatten(img, 1)
    return img


def _grouped_data_to_datamodule(
    trn: List[th.tensor],
    other_data_mapping: Dict[str, List[th.tensor]],
    batch_size: int,
) -> DataModule:
    random.shuffle(trn)
    val, tst = [], []
    constituent_data_mapping = {}
    for k, other_data in other_data_mapping.items():
        random.shuffle(other_data)
        len_data = len(other_data)
        val_count = math.ceil(len_data * 0.5)
        _val, _tst = other_data[:val_count], other_data[val_count:]
        val.extend(_val)
        tst.extend(_tst)
        constituent_data_mapping[k] = _tst
    return DataModule(batch_size, trn, val, tst, constituent_data_mapping)


def _grouped_data_to_datamodule_matrix(
    train,
    train_y,
    other_data_mapping,
    batch_size,
):
    val, tst = [], []
    val_y, tst_y = [], []
    constituent_data_mapping = {}
    for k, other_data in other_data_mapping.items():
        x, y = other_data
        # Sample half the indices for test, leave the rest for validation.
        t_idx = random.sample(list(range(len(y))), len(y) // 2)
        v_idx = [i for i in range(len(y)) if i not in t_idx]
        # Index x and y.
        t_x, t_y = x[t_idx], y[t_idx]
        v_x, v_y = x[v_idx], y[v_idx]
        # Store the test data per class for analysis.
        constituent_data_mapping[k] = MatrixDataset(t_x, t_y)
        val.append(v_x)
        val_y.append(v_y)
        tst.append(t_x)
        tst_y.append(t_y)
    vx, vy = th.cat(val), th.cat(val_y)
    # shuffle_mask = np.arange(len(vy))
    # np.random.shuffle(shuffle_mask)
    # vx, vy = vx[shuffle_mask], vy[shuffle_mask]
    tx, ty = th.cat(tst), th.cat(tst_y)
    # shuffle_mask = np.arange(len(ty))
    # np.random.shuffle(shuffle_mask)
    # tx, ty = tx[shuffle_mask], ty[shuffle_mask]
    return DataModule(
        batch_size,
        MatrixDataset(train, train_y),
        MatrixDataset(vx, vy),
        MatrixDataset(tx, ty),
        constituent_data_mapping,
    )
