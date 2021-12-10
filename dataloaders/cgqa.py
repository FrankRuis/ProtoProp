import torch.utils.data as tdata
from dataloaders.ut_zappos import CompositionDataset
from model.graph import CompositionalGraph


def get_dataloaders(batch_size=128, workers=0, undirected=False, ao_edges=False):
    train_dataset = CompositionDataset(root='data/cgqa', phase='train', split='compositional-split-natural')
    val_dataset = CompositionDataset(root='data/cgqa', phase='val', split='compositional-split-natural')
    test_dataset = CompositionDataset(root='data/cgqa', phase='test', split='compositional-split-natural')
    seen = {(train_dataset.attr2idx[a], train_dataset.obj2idx[b]) for a, b in train_dataset.train_pairs}

    attributes = train_dataset.attrs
    objects = train_dataset.objs
    compositions = train_dataset.pairs
    graph = CompositionalGraph(attributes, objects, compositions, undirected=undirected, ao_edges=ao_edges)

    train_loader = tdata.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=workers)
    validation_loader = tdata.DataLoader(val_dataset, batch_size=batch_size, num_workers=workers)
    test_loader = tdata.DataLoader(test_dataset, batch_size=batch_size, num_workers=workers)

    dataloaders = {
        'train': train_loader,
        'val': validation_loader,
        'test': test_loader
    }

    data_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset),
        'test': len(test_dataset)
    }

    return dataloaders, data_sizes, graph, seen, val_dataset
