# ProtoProp
Code for the NeurIPS 2021 paper [Independent Prototype Propagation for Zero-Shot Compositionality](https://arxiv.org/abs/2106.00305).

To train and evaluate the model change the parameters in `run_args/args.json` and run `train/train_protoprop.py`, or change the parameters in `train/grid_search.py` and run for a grid search or results with error bars.
The data can be obtained from the repositories in the attribution section below.

NOTE: ImageNet transforms (as used in `dataloaders/ut_zappos.py`) crop out significant portions of the objects depicted in UT-Zappos and C-GQA images.
We use them for fair comparison with previous works, but we would recommend using different val and test crops otherwise.

## Attribution
Parts of the code has been taken or adapted from the following sources.

Evaluation & loading UT-Zappos (these repositories also contain dowload links for the data):  
https://github.com/Tushar-N/attributes-as-operators
https://github.com/ExplainableML/czsl

HSIC calculation:  
https://github.com/choasma/HSIC-bottleneck

## Reference

    @inproceedings{ruis2021protoprop,
      title={Independent Prototype Propagation for Zero-Shot Compositionality},
      author={Ruis, Frank and Burghouts, Gertjan J and Bucur, Doina},
      booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
      volume={34},
      year={2021}
    }
