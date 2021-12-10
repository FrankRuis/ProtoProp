import torch
import torch.nn as nn
from dataloaders import get_embeddings
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SGConv


class GraphModel(nn.Module):
    def __init__(self, backbone, attr_head, obj_head, dataset, back_out, dim_proto, dim_hidden=2048, dim_out=512, dropout=0, dset=None, glayers=1, init_protos=False):
        super(GraphModel, self).__init__()
        self.dropout = dropout
        self.dset = dset
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.attr_head = attr_head
        self.obj_head = obj_head
        self.dim_proto = dim_proto
        self.protos = None

        # Initialize prototypes with word embeddings (if enabled)
        if init_protos:
            embeddings = get_embeddings(dataset).cuda()
            apn = attr_head.prototypes.size(0)
            opn = obj_head.prototypes.size(0)
            attr_head.prototypes = torch.nn.Parameter(embeddings[-(apn+opn):-opn].unsqueeze(-1).unsqueeze(-1))
            obj_head.prototypes = torch.nn.Parameter(embeddings[-opn:].unsqueeze(-1).unsqueeze(-1))
            self.protos = torch.nn.Parameter(embeddings[:len(embeddings) - (apn + opn)])

        self.glayers = glayers
        if back_out != dim_out:
            self.proj = nn.Linear(back_out, dim_out)
            self.bn = nn.BatchNorm1d(dim_out)
        else:
            self.proj = None

        if self.glayers == 1:
            self.gcn1 = GCNConv(dim_proto, dim_out)
        else:
            self.gcn1 = GCNConv(dim_proto, dim_hidden)
        self.gcn2 = GCNConv(dim_hidden, dim_out)

    def forward(self, x, attr_truth=None, obj_truth=None, pair_truth=None, graph=None, args=None):
        x = self.backbone(x)
        xc = self.pool(x).flatten(1)
        labels = [attr_truth, obj_truth]
        if self.proj is not None:
            xc = self.proj(xc)
            xc = F.relu(xc)
            xc = self.bn(xc)

        loss = 0.
        if self.training:
            for head in [self.attr_head, self.obj_head]:
                sims, outputs, distances = head(x)
                _, preds = torch.max(sims, 1)

                if attr_truth is not None:
                    loss += head.get_loss(sims, outputs, distances, graph, args, labels)
            loss /= 2

        ppc = 1
        attr_protos = torch.cat([*self.attr_head.prototypes.view((self.attr_head.prototypes.size(0) // ppc, ppc, -1)).permute((1, 0, -1))], dim=1)
        obj_protos = torch.cat([*self.obj_head.prototypes.view((self.obj_head.prototypes.size(0) // ppc, ppc, -1)).permute((1, 0, -1))], dim=1)
        if self.protos:
            embeddings = torch.cat([self.protos, attr_protos, obj_protos])
        else:
            embeddings = torch.cat([torch.zeros(len(graph.compositions), self.dim_proto * ppc).cuda(), attr_protos, obj_protos])

        if self.dropout > 0:
            embeddings = F.dropout(embeddings, self.dropout, training=self.training)

        embeddings = self.gcn1(embeddings, graph.edge_index.cuda())
        if self.glayers > 1 and not isinstance(self.gcn1, SGConv):
           if self.dropout > 0:
               embeddings = F.dropout(embeddings, self.dropout, training=self.training)
           embeddings = self.gcn2(embeddings, graph.edge_index.cuda())

        if args['metric'] == 'cosine':
            xc = xc / xc.norm(dim=1, keepdim=True)
            emb = embeddings[:len(graph.compositions)] / embeddings[:len(graph.compositions)].norm(dim=1, keepdim=True)
            xc = xc @ emb.t()
        else:
            xc = xc @ embeddings[:len(graph.compositions)].t()

        preds = {}
        for itr, pair in enumerate(self.dset.pairs):
            preds[pair] = xc[:, self.dset.pair2idx[pair]]

        if attr_truth is not None:
            loss += args['l_comp'] * F.cross_entropy(xc, pair_truth)

        return loss, preds
