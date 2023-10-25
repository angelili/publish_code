from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, Dataset, Subset, random_split, DataLoader
from torchvision.datasets import MNIST







def download_data() -> Tuple[Dataset, Dataset]:
    """Download (if necessary) and returns the MNIST dataset.

    Returns
    -------
    Tuple[MNIST, MNIST]
        The dataset for training and the dataset for testing MNIST.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    trainset = MNIST("./dataset", train=True, download=True, transform=transform)
    testset = MNIST("./dataset", train=False, download=True, transform=transform)
    return trainset, testset

def sort_by_class(trainset: Dataset) -> Dataset:
    """Sort dataset by class/label.

    Parameters
    ----------
    trainset : Dataset
        The training dataset that needs to be sorted.

    Returns
    -------
    Dataset
        The sorted training dataset.
    """
    class_counts = np.bincount(trainset.targets)   #numpy array
    idxs = trainset.targets.argsort()  # sort targets in ascending order

    tmp = []  # create subset of smallest class
    tmp_targets = []  # same for targets

    start = 0
    for count in np.cumsum(class_counts): #iterate through the cumulative sum of class counts
        tmp.append(
            Subset(trainset, idxs[start : int(count + start)])
        )  # add rest of classes
        tmp_targets.append(trainset.targets[idxs[start : int(count + start)]])
        start += count
    sorted_dataset = ConcatDataset(tmp)  # concat dataset
    sorted_dataset.targets = torch.cat(tmp_targets)  # concat targets
    
    return sorted_dataset



def power_law_split(
    sorted_trainset: Dataset,
    num_partitions: int,
    num_labels_per_partition: int = 2,
    min_data_per_partition: int = 10,
    mean: float = 0.0,
    sigma: float = 2.0,
) -> List[Dataset]:
    """Partition the dataset following a power-law distribution.

    Parameters
    ----------
    sorted_trainset : Dataset
        The training dataset sorted by label/class.
    num_partitions: int
        Number of partitions to create.
    num_labels_per_partition: int
        Number of labels to have in each dataset partition.
    min_data_per_partition: int
        Minimum number of datapoints included in each partition.
    mean: float
        Mean value for LogNormal distribution to construct power-law.
    sigma: float
        Sigma value for LogNormal distribution to construct power-law.

    Returns
    -------
    List[Dataset]
        The partitioned training datasets.
    """
    targets = sorted_trainset.targets
    full_idx = list(range(len(targets)))

    class_counts = np.bincount(sorted_trainset.targets)
    labels_cs = np.cumsum(class_counts)
    labels_cs = [0] + labels_cs[:-1].tolist()  #used to determine where each class begins and ends in the sorted dataset

    partitions_idx: List[List[int]] = [] #This list will be used to store the indices of samples that belong to each partition.
    num_classes = len(np.bincount(targets))
    hist = np.zeros(num_classes, dtype=np.int32)

    # assign min_data_per_partition
    min_data_per_class = int(min_data_per_partition / num_labels_per_partition)
    for u_id in range(num_partitions):
        partitions_idx.append([])
        for cls_idx in range(num_labels_per_partition): # select samples for different class labels 
            cls = (u_id + cls_idx) % num_classes # Calculate a range of indices to select samples from the current class
            indices = list(
                full_idx[
                    labels_cs[cls]
                    + hist[cls] : labels_cs[cls]
                    + hist[cls]
                    + min_data_per_class
                ]
            )
            partitions_idx[-1].extend(indices) # Add the selected indices to the current partition
            hist[cls] += min_data_per_class

    # add remaining images following power-law
    probs = np.random.lognormal(
        mean,
        sigma,
        (num_classes, int(num_partitions / num_classes), num_labels_per_partition),
    )
    remaining_per_class = class_counts - hist

    # obtain how many samples each partition should be assigned for each of the labels it contains
    probs = (
        remaining_per_class.reshape(-1, 1, 1)
        * probs
        / np.sum(probs, (1, 2), keepdims=True)
    )

    for u_id in range(num_partitions):
        for cls_idx in range(num_labels_per_partition):
            cls = (u_id + cls_idx) % num_classes
            count = int(probs[cls, u_id // num_classes, cls_idx])

            # add count of specific class to partition
            indices = full_idx[
                labels_cs[cls] + hist[cls] : labels_cs[cls] + hist[cls] + count
            ]
            partitions_idx[u_id].extend(indices)
            hist[cls] += count

    # construct subsets
    partitions = [Subset(sorted_trainset, p) for p in partitions_idx]

    return partitions


trainset, testset = download_data()

def load_datasets(  # pylint: disable=too-many-arguments
    num_clients: int,
    val_ratio: float = 0.1,
    batch_size: Optional[int] = 32,
    seed: Optional[int] = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Split each partition into train/val and create DataLoader

    trainset_sorted = sort_by_class(trainset)
    datasets = power_law_split(
    trainset_sorted,
    num_partitions=num_clients,
    num_labels_per_partition=2,
    min_data_per_partition=10,
    mean=0.0,
    sigma=2.0)
    trainloaders = []
    valloaders = []
    for dataset in datasets:
        len_val = int(len(dataset) / (1 / val_ratio))
        lengths = [len(dataset) - len_val, len_val]
        ds_train, ds_val = random_split(
            dataset, lengths, torch.Generator().manual_seed(seed)
        )
        trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size))

    return trainloaders, valloaders, DataLoader(testset, batch_size=batch_size)