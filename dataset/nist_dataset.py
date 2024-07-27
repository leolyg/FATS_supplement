import pathlib
from logging import INFO
from typing import List, Tuple
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch
import pandas as pd

from torch.utils.data import Dataset


class NISTLikeDataset(Dataset):
    """Dataset representing NIST or preprocessed variant of it."""

    def __init__(
        self,
        image_paths: List[pathlib.Path],
        labels: np.ndarray,
        method: str = 'base',
        transform: transforms = transforms.ToTensor(),
    ) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.method = method
        self.transform = transforms.ToTensor() if transform is None else transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label)
        if self.method == 'FATS':
            return image, label, index
        else:
            return image, label


def create_dataset(df_info: pd.DataFrame, labels: np.ndarray, method: str = 'base') -> NISTLikeDataset:
    """Instantiate NISTLikeDataset.

    Parameters
    ----------
    df_info: pd.DataFrame
        contains paths to images
    labels: np.ndarray
        0 till N-1 classes labels in the same order as in df_info

    Returns
    -------
    nist_like_dataset: NISTLikeDataset
        created dataset
    """
    nist_like_dataset = NISTLikeDataset(
        df_info["path"].values, labels, method=method)
    return nist_like_dataset


def create_partition_list(df_info: pd.DataFrame) -> List[List[int]]:
    """
    Create list of list with int masks identifying writers.
    Parameters
    ----------
    df_info: pd.DataFrame
        contains writer_id information

    Returns
    -------
    division_list: List[List[int]]
        List of lists of indices to identify unique writers
    """
    writers_ids = df_info["writer_id"].values
    unique_writers = np.unique(writers_ids)
    indices = {
        writer_id: np.where(writers_ids == writer_id)[0].tolist()
        for writer_id in unique_writers
    }
    return list(indices.values())
