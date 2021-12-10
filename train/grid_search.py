from pathlib import Path
from train.train_protoprop import main
from eval import Evaluator
import json
from itertools import product
import numpy as np
import torch

if __name__ == '__main__':
    """
    Add parameters as a list of options for a grid search of all combinations
    Parameters in a list are added to the filename, add as a list with single element to add to filename without 
    multiple options in the grid search.
    """
    repeats = 5  # number training runs (for error bars)
    options = {
        "img_size": 224,
        "batch_size": 128,
        "dim_back": 512,
        "dim_out": 512,
        "dim_proto": [256],
        "init_protos": False,
        "dim_hidden": [512],
        "undirected": True,
        "ao_edges": True,
        "g_layers": [2],  # 1 or 2
        "epochs": 35,
        "lr": [5E-5],
        "lr_comp": [5E-5],
        "l_hsic": [10],
        "l_clst": [0.01],
        "l_sep": [0.01],
        "dropout": 0,
        "obj_only_sep": True,
        "l_comp": 1,
        "metric": "dot",
        "pooling": False,
        "dataset": "zappos",
        "clevr_split": 4000,
        "clevr_seed": 0,
        "shared_proj": False,
        "optimizer": ["adam"],
        "weight_decay": 5E-5,
        "momentum": 0.9,
        "nesterov": False,
        "scheduler": False,
        "gamma": 0.5,
        "step_size": 10,
        "save_every_best": False,
        "workers": 8
    }

    out_dir = "out/grid/{}".format(options['dataset'])
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    to_search = []
    for o in options:
        if isinstance(options[o], list) and o != 'pooling':
            to_search.append([(o, v) for v in options[o]])

    for i, s in enumerate(product(*to_search)):
        print('Checking {} {} times'.format(str(s), repeats))
        cur_opts = []
        for k, v in s:
            cur_opts.append('{}={}'.format(k, v))
            options[k] = v

        acc_vals = []
        acc_tests = []
        epochs = []
        for _ in range(repeats):
            model, dataloaders, data_sizes, graph, seen, valdata = main(options)
            valevaluator = Evaluator(valdata, None)
            valdata.phase = 'test'
            testevaluator = Evaluator(valdata, None)

            accuracies, all_sub_gt, all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], [], [], []
            for data in dataloaders['val']:
                data = [d.to('cuda') for d in data]
                inputs, attr_truth, obj_truth, pair_truth = data[:4]

                with torch.set_grad_enabled(False):
                    loss, preds = model(inputs, attr_truth, obj_truth, pair_truth, graph, options)

                all_pred.append(preds)
                all_attr_gt.append(attr_truth)
                all_obj_gt.append(obj_truth)
                all_pair_gt.append(pair_truth)

            all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt), torch.cat(all_obj_gt), torch.cat(all_pair_gt)

            all_pred_dict = {}
            # Gather values as dict of (attr, obj) as key and list of predictions as values
            for k in all_pred[0].keys():
                all_pred_dict[k] = torch.cat(
                    [all_pred[i][k] for i in range(len(all_pred))])

            # Calculate best unseen accuracy
            results = valevaluator.score_model(all_pred_dict, all_obj_gt, bias=1000, topk=1)
            stats = valevaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict,
                                                      topk=1)
            acc_vals.append(
                np.array([stats['hm_seen'], stats['hm_unseen'], stats['best_hm'], stats['AUC'], stats['biasterm'], stats['best_seen'], stats['best_unseen']]))

            accuracies, all_sub_gt, all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], [], [], []
            for data in dataloaders['test']:
                data = [d.to('cuda') for d in data]
                inputs, attr_truth, obj_truth, pair_truth = data[:4]

                with torch.set_grad_enabled(False):
                    loss, preds = model(inputs, attr_truth, obj_truth, pair_truth, graph, options)

                all_pred.append(preds)
                all_attr_gt.append(attr_truth)
                all_obj_gt.append(obj_truth)
                all_pair_gt.append(pair_truth)

            all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt), torch.cat(all_obj_gt), torch.cat(all_pair_gt)
            all_pred_dict = {}
            # Gather values as dict of (attr, obj) as key and list of predictions as values
            for k in all_pred[0].keys():
                all_pred_dict[k] = torch.cat(
                    [all_pred[i][k] for i in range(len(all_pred))])

            # Calculate best unseen accuracy
            results = testevaluator.score_model(all_pred_dict, all_obj_gt, bias=1000, topk=1)
            stats = testevaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict,
                                                      topk=1)
            acc_tests.append(
                np.array([stats['hm_seen'], stats['hm_unseen'], stats['best_hm'], stats['AUC'], stats['biasterm'], stats['best_seen'], stats['best_unseen']]))

        mn_vals, std_vals = np.mean(acc_vals, axis=0), np.std(acc_vals, axis=0)
        mn_tests, std_tests = np.mean(acc_tests, axis=0), np.std(acc_tests, axis=0)

        with open(out_dir + '/{:.4f}+-{:.2f}%-{}.txt'.format(mn_tests[2], std_tests[2] * 100, '+'.join(cur_opts)), 'w') as file:
            file.writelines([
                json.dumps(options) + '\n',
                '\n',
                'Val:\n',
                'Acc Seen: {:.4f}+-{:.2f}%\n'.format(mn_vals[0], std_vals[0] * 100),
                'Acc Unseen: {:.4f}+-{:.2f}%\n'.format(mn_vals[1], std_vals[1] * 100),
                'HM: {:.4f}+-{:.2f}%\n'.format(mn_vals[2], std_vals[2] * 100),
                'AUC: {:.4f}+-{:.2f}%\n'.format(mn_vals[3], std_vals[3] * 100),
                'Bias: {}+-{}\n'.format(np.round(mn_vals[4]), np.round(std_vals[4])),
                'Closed Seen: {:.4f}+-{:.2f}%\n'.format(mn_vals[5], std_vals[5] * 100),
                'Closed Unseen: {:.4f}+-{:.2f}%\n'.format(mn_vals[6], std_vals[6] * 100),
                '\n',
                'Test:\n',
                'Acc Seen: {:.4f}+-{:.2f}%\n'.format(mn_tests[0], std_tests[0] * 100),
                'Acc Unseen: {:.4f}+-{:.2f}%\n'.format(mn_tests[1], std_tests[1] * 100),
                'HM: {:.4f}+-{:.2f}%\n'.format(mn_tests[2], std_tests[2] * 100),
                'AUC: {:.4f}+-{:.2f}%\n'.format(mn_tests[3], std_tests[3] * 100),
                'Bias: {}+-{}\n'.format(np.round(mn_tests[4]), np.round(std_tests[4])),
                'Closed Seen: {:.4f}+-{:.2f}%\n'.format(mn_tests[5], std_tests[5] * 100),
                'Closed Unseen: {:.4f}+-{:.2f}%\n'.format(mn_tests[6], std_tests[6] * 100)
            ])
