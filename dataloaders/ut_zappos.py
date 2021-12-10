# File adapted from https://github.com/Tushar-N/attributes-as-operators

import numpy as np
import torch.utils.data as tdata
import torchvision.transforms as transforms
from PIL import Image
import torch
from model.graph import CompositionalGraph


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


class ImageLoader:
    def __init__(self, root):
        self.img_dir = root

    def __call__(self, img):
        file = '%s/%s' % (self.img_dir, img)
        img = Image.open(file).convert('RGB')
        return img


def imagenet_transform(phase):
    """
    Warning, imagenet transforms cut out significant portions of UT-Zappos and C-GQA images.
    We use them for fair comparison with previous works, but we would recommend using different val and test crops otherwise.
    """
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if phase == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop((224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif phase == 'test' or phase == 'val':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    return transform


#------------------------------------------------------------------------------------------------------------------------------------#


class CompositionDataset(tdata.Dataset):
    def __init__(
            self,
            root,
            phase,
            split='compositional-split',
            subset=False,
            num_negs=1,
            pair_dropout=0.0,
    ):
        self.root = root
        self.phase = phase
        self.split = split
        self.num_negs = num_negs
        self.pair_dropout = pair_dropout

        self.feat_dim = None
        self.transform = imagenet_transform(phase)
        self.loader = ImageLoader(self.root + '/images/')

        self.attrs, self.objs, self.pairs, \
                self.train_pairs, self.val_pairs, \
                self.test_pairs = self.parse_split()

        self.train_data, self.val_data, self.test_data = self.get_split_info()
        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        else:
            self.data = self.test_data
        if subset:
            ind = np.arange(len(self.data))
            ind = ind[::len(ind) // 1000]
            self.data = [self.data[i] for i in ind]

        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        print('# train pairs: %d | # val pairs: %d | # test pairs: %d' % (len(
            self.train_pairs), len(self.val_pairs), len(self.test_pairs)))
        print('# train images: %d | # val images: %d | # test images: %d' %
              (len(self.train_data), len(self.val_data), len(self.test_data)))

        # fix later -- affordance thing
        # return {object: all attrs that occur with obj}
        self.obj_affordance = {}
        self.train_obj_affordance = {}
        for _obj in self.objs:
            candidates = [
                attr
                for (_, attr,
                     obj) in self.train_data + self.val_data + self.test_data
                if obj == _obj
            ]
            self.obj_affordance[_obj] = sorted(list(set(candidates)))

            candidates = [
                attr for (_, attr, obj) in self.train_data if obj == _obj
            ]
            self.train_obj_affordance[_obj] = sorted(list(set(candidates)))

        self.sample_indices = list(range(len(self.data)))
        self.sample_pairs = self.train_pairs

    def reset_dropout(self):
        self.sample_indices = list(range(len(self.data)))
        self.sample_pairs = self.train_pairs

        shuffled_ind = np.random.permutation(len(self.train_pairs))
        n_pairs = int((1 - self.pair_dropout) * len(self.train_pairs))
        self.sample_pairs = [
            self.train_pairs[pi] for pi in shuffled_ind[:n_pairs]
        ]
        print('Using {} pairs out of {} pairs right now'.format(
            n_pairs, len(self.train_pairs)))
        self.sample_indices = [
            i for i in range(len(self.data))
            if (self.data[i][1], self.data[i][2]) in self.sample_pairs
        ]
        print('Using {} images out of {} images right now'.format(
            len(self.sample_indices), len(self.data)))

    def get_split_info(self):
        data = torch.load(self.root + '/metadata_{}.t7'.format(self.split))
        train_data, val_data, test_data = [], [], []
        for instance in data:
            image, attr, obj, settype = instance['image'], instance[
                'attr'], instance['obj'], instance['set']

            if attr == 'NA' or (attr,
                                obj) not in self.pairs or settype == 'NA':
                # ignore instances with unlabeled attributes
                # ignore instances that are not in current split
                continue

            data_i = [image, attr, obj]
            if settype == 'train':
                train_data.append(data_i)
            elif settype == 'val':
                val_data.append(data_i)
            else:
                test_data.append(data_i)

        return train_data, val_data, test_data

    def parse_split(self):
        def parse_pairs(pair_list):
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                # pairs = [t.split() if not '_' in t else t.split('_') for t in pairs]
                pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            '%s/%s/train_pairs.txt' % (self.root, self.split))
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            '%s/%s/val_pairs.txt' % (self.root, self.split))
        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            '%s/%s/test_pairs.txt' % (self.root, self.split))

        all_attrs, all_objs = sorted(
            list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(
                list(set(tr_objs + vl_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

    def sample_negative(self, attr, obj):
        new_attr, new_obj = self.sample_pairs[np.random.choice(
            len(self.sample_pairs))]
        if new_attr == attr and new_obj == obj:
            return self.sample_negative(attr, obj)
        return (self.attr2idx[new_attr], self.obj2idx[new_obj])

    def sample_affordance(self, attr, obj):
        new_attr = np.random.choice(self.obj_affordance[obj])
        if new_attr == attr:
            return self.sample_affordance(attr, obj)
        return self.attr2idx[new_attr]

    def sample_train_affordance(self, attr, obj):
        new_attr = np.random.choice(self.train_obj_affordance[obj])
        if new_attr == attr:
            return self.sample_train_affordance(attr, obj)
        return self.attr2idx[new_attr]

    def __getitem__(self, index):
        index = self.sample_indices[index]
        image, attr, obj = self.data[index]
        img = self.loader(image)
        img = self.transform(img)

        data = [
            img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]
        ]

        return data

    def __len__(self):
        return len(self.sample_indices)


def get_dataloaders(batch_size=256, workers=1, undirected=False, ao_edges=False):
    print('Don\'t forget to fix ImageNet transforms clipping.')
    train_dataset = CompositionDataset(root='data/ut-zap50k', phase='train', split='compositional-split-natural')
    val_dataset = CompositionDataset(root='data/ut-zap50k', phase='val', split='compositional-split-natural')
    test_dataset = CompositionDataset(root='data/ut-zap50k', phase='test', split='compositional-split-natural')
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
