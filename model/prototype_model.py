import torch
import torch.nn as nn
import torch.nn.functional as F
from model.hsic import hsic_normalized


class LocalProts(nn.Module):
    def __init__(self, ch_in, n_class, prot_dim=512, prot_shape=(1, 1), bn=True, metric='dot', pooling=False, proj=True, type_=None, i=None):
        super(LocalProts, self).__init__()
        self.type_ = type_
        self.i = i
        self.bn = bn
        self.prot_dim = prot_dim
        self.n_class = n_class
        self.prototype_shape = (self.n_class, self.prot_dim, *prot_shape)
        self.metric = metric
        self.pooling = pooling
        self.proj = proj

        self.class_identity = torch.zeros(self.prototype_shape[0], self.n_class)
        for i in range(self.class_identity.size(0)):
            self.class_identity[i, i] = 1

        self.conv1 = nn.Conv2d(ch_in, self.prot_dim, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(self.prot_dim)

        self.prototypes = nn.Parameter(torch.rand(self.prototype_shape))
        self.logit_scale = nn.Parameter(torch.ones([]))

        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)

    def forward(self, x):
        if self.pooling:
            x = F.adaptive_avg_pool2d(x, (1, 1))

        if self.proj:
            x = self._project(x)

        dist = self._distance(x)

        sm = F.softmax(dist.view(*dist.size()[:2], -1), dim=2).view_as(dist)
        vecs = []
        for i in range(sm.size(1)):
            smi = sm[:, i].unsqueeze(1)
            vecs.append(torch.mul(x, smi))
        vecs = torch.stack(vecs, dim=1).sum(dim=(3, 4))
        dist = (vecs * self.prototypes.flatten(1)).sum(2)
        print(dist.shape)

        # if self.metric == 'euclidean':
        #     # min pooling for euclidean distance
        #     dist, idxs = F.max_pool2d(-dist, dist.shape[2:], return_indices=True)
        #     dist = -dist
        # else:
        #     dist, idxs = F.max_pool2d(dist, dist.shape[2:], return_indices=True)
        # dist = dist.flatten(1)

        out = dist
        if self.metric == 'euclidean':
            out = -out

        return out, vecs, dist

    def get_loss(self, attr_sims, attr_outputs, distances, graph, args, labels, device='cuda'):
        targets = labels[self.i]

        loss = F.cross_entropy(attr_sims, targets)
        if args['l_hsic'] != 0:
            nclasses = len(graph.objects) if self.type_ == 'attributes' else len(graph.attributes)
            attr_onehots = torch.squeeze(torch.eye(nclasses)[labels[self.i - 1]])
            if attr_outputs.size(1) == self.n_class:
                attr_outputs = attr_outputs[range(attr_outputs.size(0)), targets]
            else:
                n = self.prots_per_class
                attr_outputs = torch.cat(
                    [attr_outputs[:, j::n][range(attr_outputs.size(0)), targets] for j in range(n)])
                attr_onehots = attr_onehots.repeat(n, 1)

            loss += args['l_hsic'] * hsic_normalized(attr_outputs, attr_onehots)

        if args['l_clst'] != 0:
            loss += args['l_clst'] * self.cluster_costs(distances, targets, device=device)

        if args['l_sep'] != 0 and not args['obj_only_sep']:
            loss += args['l_sep'] * self.separation_costs(distances, targets, device=device)
        elif args['l_sep'] != 0 and self.type_ == 'objects':
            loss += args['l_sep'] * self.separation_costs(distances, targets, device=device)

        return loss

    def _project(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        if self.bn:
            x = self.bn1(x)

        return x

    def _distance(self, x):
        if callable(self.metric):
            dist = self.metric(x)
        elif self.metric == 'dot':
            # dist = self.logit_scale.exp() * F.conv2d(x, weight=self.prototypes)
            dist = F.conv2d(x, weight=self.prototypes)
        elif self.metric == 'euclidean':
            dist = self._l2_convolution(x)
        elif self.metric == 'cosine':
            x = x / x.norm(dim=1, keepdim=True)
            weight = self.prototypes / self.prototypes.norm(dim=1, keepdim=True)
            dist = self.logit_scale.exp() * F.conv2d(x, weight=weight)
        else:
            raise NotImplementedError('Metric {} not implemented.'.format(self.metric))

        return dist

    def cluster_costs(self, distances, labels, device='cuda'):
        """
        Loss adapted from https://github.com/cfchen-duke/ProtoPNet
        """
        target_prots = self.class_identity[:, labels].t().to(device)
        if self.metric == 'cosine':
            mx = self.logit_scale
        elif self.metric == 'euclidean':
            mx = self.prototype_shape[1] * self.prototype_shape[2] * self.prototype_shape[3]
        else:
            mx = torch.max(distances)

        if self.metric == 'euclidean':
            max_dist_per_prot, _ = torch.max((mx - distances) * target_prots, dim=1)
        else:
            max_dist_per_prot, _ = torch.max(distances * target_prots, dim=1)

        cluster_cost = torch.mean(mx - max_dist_per_prot)

        return cluster_cost

    def separation_costs(self, distances, labels, device='cuda'):
        """
        Loss adapted from https://github.com/cfchen-duke/ProtoPNet
        """
        if self.metric == 'euclidean':
            mx = self.prototype_shape[1] * self.prototype_shape[2] * self.prototype_shape[3]

        other_prots = 1 - self.class_identity[:, labels].t().to(device)
        if self.metric == 'euclidean':
            max_dist_per_other_prot, _ = torch.max((mx - distances) * other_prots, dim=1)
            separation_cost = torch.mean(mx - max_dist_per_other_prot)
        elif self.metric == 'cosine':
            max_dist_per_other_prot, _ = torch.max(distances * other_prots, dim=1)
            separation_cost = torch.mean(self.logit_scale + max_dist_per_other_prot)
        else:
            max_dist_per_other_prot, _ = torch.max(distances * other_prots, dim=1)
            separation_cost = torch.mean(max_dist_per_other_prot)

        return separation_cost

    def _l2_convolution(self, x):
        """
        Taken from https://github.com/cfchen-duke/ProtoPNet
        apply self.prototype_vectors as l2-convolution filters on input x
        """
        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones)

        p2 = self.prototypes ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=self.prototypes)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances
