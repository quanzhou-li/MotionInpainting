import sys
sys.path.append('.')
sys.path.append('..')

import argparse
from tools.cfg_parser import Config
from train.trainer import Trainer


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='MotionInfilling-Training')

    parser.add_argument('--work-dir', required=True, type=str,
                        help='Saving path')

    parser.add_argument('--data-path', required=True, type=str,
                        help='The path to the folder that contains MNet data')

    parser.add_argument('--inr-config', required=True, type=str)

    parser.add_argument('--expr-ID', default='V00', type=str,
                        help='Training ID')

    parser.add_argument('--batch-size', default=16, type=int,
                        help='Training batch size')

    parser.add_argument('--n-workers', default=8, type=int,
                        help='Number of PyTorch dataloader workers')

    parser.add_argument('--n-epochs', default=200, type=int,
                        help='Number of epochs')

    parser.add_argument('--train-data', default='train_data', type=str,
                        help='Name of the training set')

    parser.add_argument('--val-data', default='val_data', type=str,
                        help='Name of the validation set')

    parser.add_argument('--lr', default=5e-4, type=float,
                        help='Training learning rate')

    parser.add_argument('--kl-coef', default=1e-2, type=float,
                        help='KL divergence coefficent for GNet training')

    parser.add_argument('--use-multigpu', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='If to use multiple GPUs for training')

    parser.add_argument('--best-model', default=None, type=str,
                        help='path to the best model')

    args = parser.parse_args()

    work_dir = args.work_dir
    data_path = args.data_path
    expr_ID = args.expr_ID
    n_workers = args.n_workers
    use_multigpu = args.use_multigpu
    batch_size = args.batch_size
    base_lr = args.lr
    kl_coef = args.kl_coef
    n_epochs = args.n_epochs
    ds_train = args.train_data
    ds_val = args.val_data
    inr_config = args.inr_config
    best_model = args.best_model

    cfg = {
        'work_dir': work_dir,
        'ds_train': ds_train,
        'ds_val': ds_val,
        'inr_config': inr_config,
        'batch_size': batch_size,
        'n_workers': n_workers,
        'use_multigpu': use_multigpu,
        'kl_coef': kl_coef,
        'dataset_dir': data_path,
        'expr_ID': expr_ID,
        'base_lr': base_lr,
        'n_epochs': n_epochs,
        'save_every_it': 10,
        'best_model': best_model,
        'try_num': 0,
        'reg_coef': 5e-4
    }

    cfg = Config(**cfg)
    trainer = Trainer(cfg=cfg)

    trainer.fit()