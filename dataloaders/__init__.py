from dataloaders.ao_clevr import get_dataloaders
from dataloaders.ut_zappos import get_dataloaders as zappos_dataloaders
from dataloaders.cgqa import get_dataloaders as cgqa_dataloaders

import torchvision.transforms as T
import torch


def get_dataloader(args):
    transforms = T.Compose([
        T.Resize(args['img_size']),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if args['dataset'] == 'clevr':
        dataloaders, data_sizes, graph, seen, valdata = get_dataloaders(transforms, batch_size=args['batch_size'], spl='VT', seed=args['clevr_split'],
                                                         seen_seed=args['clevr_seed'], workers=args['workers'], undirected=args['undirected'], ao_edges=args['ao_edges'])
    elif args['dataset'] == 'zappos':
        dataloaders, data_sizes, graph, seen, valdata = zappos_dataloaders(batch_size=args['batch_size'], workers=args['workers'], undirected=args['undirected'], ao_edges=args['ao_edges'])
    elif args['dataset'] == 'cgqa':
        dataloaders, data_sizes, graph, seen, valdata = cgqa_dataloaders(batch_size=args['batch_size'], workers=args['workers'], undirected=args['undirected'], ao_edges=args['ao_edges'])
    else:
        raise ValueError('Unknown dataset {}.'.format(args['dataset']))

    return dataloaders, data_sizes, graph, seen, valdata


def get_embeddings(dataset):
    if dataset == 'clevr':
        return torch.load('data/ao_clevr_word_embeddings.pt')
    elif dataset == 'zappos':
        return torch.load('data/zappos_word_embeddings.pt')
    elif dataset == 'cgqa':
        # Use the embeddings provided by Naeem et al. https://github.com/ExplainableML/czsl
        g = torch.load('czsl-main/utils/cgqa-graph.t7')
        return torch.cat([g['embeddings'][-9378:], g['embeddings'][:-9378]])
    else:
        raise ValueError('No embeddings found for dataset {}.'.format(dataset))
