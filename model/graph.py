import torch


class CompositionalGraph:
    def __init__(self, attributes, objects, compositions, undirected=False, seen=None, ao_edges=True):
        ppc = 1
        self.attributes = [a for e in attributes for a in [e + str(i) if i > 0 else e for i in range(ppc)]]
        self.objects = [o for e in objects for o in [e + str(i) if i > 0 else e for i in range(ppc)]]
        self.compositions = compositions
        self.nodes = self.compositions + self.attributes + self.objects
        self.ntoi = {e: i for i, e in enumerate(self.nodes)}
        self.ppc = ppc

        if seen:
            sattr = set([e[0] for e in seen])
            sobj = set([e[1] for e in seen])
            skattr = {e for e in self.attributes if e not in sattr}
            skobj = {e for e in self.objects if e not in sobj}

        edges = []
        relations = []
        for a, o in compositions:
            if seen is not None and (a in skattr or o in skobj):
                continue

            # edges from attributes and objects to compositional nodes
            edges.append((self.ntoi[a], self.ntoi[(a, o)]))
            relations.append(0)
            edges.append((self.ntoi[o], self.ntoi[(a, o)]))
            relations.append(1)

            if ppc > 1:
                for i in range(1, ppc):
                    edges.append((self.ntoi[a + str(i)], self.ntoi[(a, o)]))
                    relations.append(0)
                    edges.append((self.ntoi[o + str(i)], self.ntoi[(a, o)]))
                    relations.append(1)

            # undirected edges between attributes and objects
            if ao_edges:
                edges.append((self.ntoi[a], self.ntoi[o]))
                relations.append(4)
                edges.append((self.ntoi[o], self.ntoi[a]))
                relations.append(5)

            # add reverse edges
            if undirected:
                edges.append((self.ntoi[(a, o)], self.ntoi[a]))
                relations.append(2)
                edges.append((self.ntoi[(a, o)], self.ntoi[o]))
                relations.append(3)

        self.edge_index = torch.cat((torch.tensor([e[0] for e in edges]).unsqueeze(0), torch.tensor([e[1] for e in edges]).unsqueeze(0)))
        self.relations = torch.tensor(relations, dtype=torch.long)

    def get_target(self, ai, oi):
        return self.ntoi[(self.attributes[ai * self.ppc], self.objects[oi * self.ppc])]
