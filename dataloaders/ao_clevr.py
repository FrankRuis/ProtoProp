import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor
from torch.utils.data.sampler import SubsetRandomSampler
from random import sample
import pickle
from model.graph import CompositionalGraph


class AOClevr(Dataset):
    """
    AO-Clevr dataset.
    Returns (image, (color, shape, material)) tuples.
    """

    def __init__(self, root_dir='data/images', csv_file='data/objects_metadata.csv', transform=Compose([lambda x: x.convert('RGB'), ToTensor()]), one_hot=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.metadata = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.attributes = ['color', 'shape', 'material', 'size']
        self.one_hot = one_hot
        self.one_hots = {}

        for a in self.attributes:
            self.metadata[a] = self.metadata[a].astype('category')
            self.one_hots[a] = torch.eye(len(self.metadata[a].unique()))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.metadata.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)

        if self.one_hot:
            attributes = [self.one_hots[a][self.metadata[a].cat.codes.iloc[idx]] for a in self.attributes]
        else:
            attributes = np.array([self.metadata[a].cat.codes.iloc[idx] for a in self.attributes])

        # return image, attributes
        return [image, *torch.LongTensor(attributes[:2]), self.pair2idx[(self.attrs[attributes[0]], self.objs[attributes[1]])]]

    def get_attribute_name(self, idx):
        return self.attributes[idx]

    def get_value_name(self, idx, val):
        return self.metadata[self.attributes[idx]].cat.categories[val]


def get_data_stats(dataset=None):
    if not dataset:
        dataset = AOClevr(transform=Compose([lambda x: x.convert('RGB'), ToTensor()]))

    sampler = SubsetRandomSampler(sample(list(range(len(dataset))), len(dataset)//10))
    means = []
    stds = []
    for idx in sampler:
        means.append(torch.mean(dataset[idx][0], dim=(-1, 1)).numpy())
        stds.append(torch.std(dataset[idx][0], dim=(-1, 1)).numpy())

    mean = torch.mean(torch.tensor(means), dim=0)
    std = torch.mean(torch.tensor(stds), dim=0)

    return mean, std


def get_dataloaders(transforms, batch_size=512, spl='VT', seed=4000, seen_seed=0, workers=1, undirected=False, ao_edges=False):
    filename = 'data/metadata_pickles/metadata_ao_clevr__{}_random__comp_seed_{}__seen_seed_{}__train.pkl'.format(
        spl,
        seed,
        seen_seed
    )
    with open(filename, 'rb') as file:
        split = pickle.load(file)

    dataset = AOClevr(transform=transforms)
    attributes = list(split['attrs'])
    objects = list(split['objs'])
    compositions = list(split['pair2idx'].keys())

    dataset.attrs = attributes
    dataset.objs = objects
    dataset.pairs = compositions

    seen = list(split['seen_pairs'])
    seen_ids = {(attributes.index(a), objects.index(o)) for (a, o) in seen}
    unseen_val = list(split['unseen_closed_val_pairs'])
    unseen_test = list(split['unseen_closed_test_pairs'])

    graph = CompositionalGraph(attributes, objects, compositions, undirected=undirected, seen=seen, ao_edges=ao_edges)
    dataset.attr2idx = {a: graph.ntoi[a] - len(compositions) for a in attributes}
    dataset.obj2idx = {o: graph.ntoi[o] - len(compositions) - len(attributes) for o in objects}
    dataset.pair2idx = graph.ntoi

    dataset.train_pairs = seen
    dataset.val_pairs = unseen_val
    dataset.test_pairs = unseen_test
    dataset.phase = 'val'

    train_indices = list(dataset.metadata.loc[dataset.metadata['image_filename'].isin(
        {e[0] for e in split['train_data'] if e[1:] in seen and e[1:] not in unseen_val})].index)
    val_indices = list(dataset.metadata.loc[dataset.metadata['image_filename'].isin(
        {e[0] for e in split['val_data'] if e[1:]})].index)
    test_indices = list(dataset.metadata.loc[dataset.metadata['image_filename'].isin(
        {e[0] for e in split['test_data'] if e[1:]})].index)

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=workers, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=workers, pin_memory=True)

    dataloaders = {
        'train': train_loader,
        'val': validation_loader,
        'test': test_loader
    }

    data_sizes = {
        'train': len(train_indices),
        'val': len(val_indices),
        'test': len(test_indices)
    }

    return dataloaders, data_sizes, graph, seen_ids, dataset
