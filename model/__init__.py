import torch


def get_optimizer(args, parameters):
    if args['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=args['lr'], weight_decay=args['weight_decay'])
    elif args['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(parameters, lr=args['lr'], momentum=args['momentum'], weight_decay=0,
                                    nesterov=args['nesterov'])
    else:
        raise ValueError('Unknown optimizer {}.'.format(args['optimizer']))

    if args['scheduler']:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['step_size'], gamma=args['gamma'])
    else:
        lr_scheduler = None

    return optimizer, lr_scheduler
