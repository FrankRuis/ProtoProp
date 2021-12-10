from dataloaders import get_dataloader
import torch
import torch.nn as nn
from torchvision.models import resnet18
from tqdm import tqdm
from model import get_optimizer
from model.skeleton import GraphModel
from model.prototype_model import LocalProts
import copy
import json
from eval import Evaluator
import torch.backends.cudnn as cudnn
cudnn.benchmark = True


def train_model(model, optimizer, lr_scheduler, dataloaders, data_sizes, graph, args, prev_acc=0, t_metric='total', device='cuda', eval=None):
    model_params = copy.deepcopy(model.state_dict())
    teval = Evaluator(dataloaders['val'].dataset, None)

    for epoch in range(args['epochs']):
        print('Starting epoch {}:'.format(epoch))

        for phase in ['train', 'val']:
            eval.dset.phase = phase
            teval.dset.phase = phase
            if phase == 'train':
                model.train()
            else:
                model.eval()

            accuracies, all_sub_gt, all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], [], [], []

            running_loss = 0.
            for data in tqdm(dataloaders[phase]) if phase == 'train' else dataloaders[phase]:
                data = [d.to(device) for d in data]
                inputs, attr_truth, obj_truth, pair_truth = data[:4]

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    loss, preds = model(inputs, attr_truth, obj_truth, pair_truth, graph, args)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                all_pred.append(preds)
                all_attr_gt.append(attr_truth)
                all_obj_gt.append(obj_truth)
                all_pair_gt.append(pair_truth)

                running_loss += (loss.item() if isinstance(loss, torch.Tensor) else loss) * inputs.size(0)

            all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt), torch.cat(all_obj_gt), torch.cat(all_pair_gt)

            all_pred_dict = {}
            # Gather values as dict of (attr, obj) as key and list of predictions as values
            for k in all_pred[0].keys():
                all_pred_dict[k] = torch.cat(
                    [all_pred[i][k] for i in range(len(all_pred))])

            # Calculate best unseen accuracy
            if phase == 'train' or phase == 'val':
                results = eval.score_model(all_pred_dict, all_obj_gt, bias=1000, topk=1)
                stats = eval.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict,
                                                       topk=1)
            else:
                results = teval.score_model(all_pred_dict, all_obj_gt, bias=1000, topk=1)
                stats = teval.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict,
                                                       topk=1)

            stats['a_epoch'] = epoch

            if phase == 'train' and lr_scheduler:
                lr_scheduler.step()

            epoch_loss = running_loss / data_sizes[phase]

            if phase == 'val' and prev_acc <= stats['best_hm']:
                print('Val accuracy higher than previous best {:.4f}'.format(prev_acc))
                prev_acc = stats['best_hm']
                model_params = copy.deepcopy(model.state_dict())

                if args['save_every_best']:
                    torch.save(model, 'out/cur_best.pt')

            if phase == 'val':
                print('{} Loss: {:.4f} Acc: {:.4f} (hm seen {:.4f}, hm unseen {:.4f}, AUC {:.4f}) @ bias {:.4f} (seen {:.4f}, unseen {:.4f})'.format(phase, epoch_loss, stats['best_hm'], stats['hm_seen'], stats['hm_unseen'], stats['AUC'], stats['biasterm'], stats['best_seen'], stats['best_unseen']))
            else:
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))

    # Load best model
    model.load_state_dict(model_params)
    print('Best acc: {:.4f}'.format(prev_acc))

    return model


def main(args):
    print(args)

    # Helps with nan detection
    torch.autograd.set_detect_anomaly(True)

    dataloaders, data_sizes, graph, seen, valdata = get_dataloader(args)

    layers = list(resnet18(pretrained=True).children())
    backbone = nn.Sequential(*layers[:-2]).cuda()

    amodel = LocalProts(args['dim_back'], len(graph.attributes), prot_dim=args['dim_proto'],
                                 prot_shape=(1, 1), metric=args['metric'], pooling=args['pooling'], proj=not args['shared_proj'], type_='attributes', i=0).cuda()
    omodel = LocalProts(args['dim_back'], len(graph.objects), prot_dim=args['dim_proto'],
                                 prot_shape=(1, 1), metric=args['metric'], pooling=args['pooling'], proj=not args['shared_proj'], type_='objects', i=1).cuda()

    model = GraphModel(backbone, amodel, omodel, args['dataset'], args['dim_back'], args['dim_proto'], args['dim_hidden'],
                       args['dim_out'], args['dropout'], dset=valdata, glayers=args['g_layers'], init_protos=args['init_protos']).cuda()
    model = torch.nn.DataParallel(model)

    print('Training model {}.'.format(model.__class__.__name__))

    parameters = [
        {'params': backbone.parameters()},
        {'params': list(set(model.parameters()) - set(backbone.parameters())), 'lr': args['lr_comp']}
    ]
    optimizer, lr_scheduler = get_optimizer(args, parameters)

    seen = {
        'attributes': {e[0] for e in seen},
        'objects': {e[1] for e in seen},
        'total': seen,
        'comp': {graph.get_target(*e) for e in seen}
    }
    evaluator = Evaluator(valdata, None)

    model = train_model(model, optimizer, lr_scheduler, dataloaders, data_sizes, graph, args, t_metric='comp', device='cuda', eval=evaluator)

    return model, dataloaders, data_sizes, graph, seen, valdata


if __name__ == '__main__':
    with open('run_args/args.json') as file:
        args = json.load(file)

    model, dataloaders, data_sizes, graph, seen, valdata = main(args)
