
# Code adapted from GrabNet

import os
import sys

import torch

sys.path.append('.')
sys.path.append('..')

from datetime import datetime
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from firelab.config import Config as FirelabCFG

from tools.utils import makepath, makelogger, CRot2rotmat
from data.dataloader import LoadData
from models.inr_gan import *


class Trainer:

    def __init__(self, cfg):

        work_dir = cfg.work_dir
        starttime = datetime.now().replace(microsecond=0)
        makepath(work_dir, isfile=False)
        logger = makelogger(makepath(os.path.join(work_dir, 'train_motion_infill.log'), isfile=True)).info
        self.logger = logger
        self.dtype = torch.float32

        summary_logdir = os.path.join(work_dir, 'summaries')
        self.swriter = SummaryWriter(log_dir=summary_logdir)
        logger(' - Started training Motion Infilling Net, experiment code %s' % (starttime))
        logger('tensorboard --logdir=%s' % summary_logdir)
        logger('Torch Version: %s\n' % torch.__version__)

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.empty_cache()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ds_train = LoadData(cfg.dataset_dir, ds_name=cfg.ds_train)
        ds_val = LoadData(cfg.dataset_dir, ds_name=cfg.ds_val)
        self.ds_train = DataLoader(ds_train, batch_size=cfg.batch_size, num_workers=cfg.n_workers, shuffle=True,
                                   drop_last=True)
        self.ds_val = DataLoader(ds_val, batch_size=cfg.batch_size, num_workers=cfg.n_workers, shuffle=True,
                                 drop_last=True)

        self.config_inr = FirelabCFG.load(cfg.inr_config)
        self.inr = INRGenerator_obj(self.config_inr).to(self.device)

        if cfg.use_multigpu:
            self.inr = nn.DataParallel(self.inr)
            logger("Training on Multiple GPUs")

        self.LossL1 = torch.nn.L1Loss(reduction='mean')
        self.LossL2 = torch.nn.MSELoss(reduction='mean')

        self.try_num = cfg.try_num
        self.epochs_completed = 0

        vars_inr = [var[1] for var in self.inr.named_parameters()]
        inr_n_params = sum(p.numel() for p in vars_inr if p.requires_grad)
        logger('Total Trainable Parameters for INR Net is %2.2f M.' % ((inr_n_params) * 1e-6))

        self.optimizer_inr = optim.Adam(vars_inr, lr=cfg.base_lr, weight_decay=cfg.reg_coef)

        self.best_loss_inr = np.inf
        self.cfg = cfg

        if cfg.best_model is not None:
            self.inr.load_state_dict(torch.load(cfg.best_model, map_location=self.device))

        torch.save(self.inr.inr.basis_matrix, os.path.join(cfg.work_dir, 'fourier_basis_matrix.pt'))

    def train(self):

        self.inr.train()
        torch.autograd.set_detect_anomaly(True)

        train_loss_dict_inr = {}

        for it, data in enumerate(self.ds_train):
            data = {k: data[k].to(self.device) for k in data.keys()}
            self.optimizer_inr.zero_grad()
            bs, _, width = data['motion_img'].shape

            obj_ori_trans = data['obj_traj_computed'][:, :, 0]
            obj_ori_orien = data['obj_orient'][:, :, 0]
            ori_ho_contact = data['ho_contact'][:, :, 0]
            drec_inr = self.inr(data['obj_traj_computed'], obj_ori_trans, obj_ori_orien, ori_ho_contact, width, 3+6+15, self.device)

            loss_total_inr, cur_loss_dict_inr = self.loss_inr(data, drec_inr)

            # loss_g, cur_loss_dict_g = self.loss_adv_g(data, drec_inr)
            # loss_total_inr += loss_g

            # loss_total_inr.backward(retain_graph=True)
            loss_total_inr.backward()

            # torch.nn.utils.clip_grad_value_(self.inr.parameters(), 5)
            self.optimizer_inr.step()

            '''self.optimizer_d.zero_grad()
            drec_inr = self.inr(data['motion_img']*mask, fframes, lframes, width, height, self.device)
            loss_d, cur_loss_dict_d = self.loss_adv_d(data, drec_inr)
            loss_d.backward()
            self.optimizer_d.step()'''

            # cur_loss_dict_inr['loss_total'] += loss_g + loss_d
            # train_loss_dict_inr = {k: train_loss_dict_inr.get(k, 0.0) + v.item() for k, v in
            #                        {**cur_loss_dict_inr, **cur_loss_dict_g, **cur_loss_dict_d}.items()}
            train_loss_dict_inr = {k: train_loss_dict_inr.get(k, 0.0) + v.item() for k, v in cur_loss_dict_inr.items()}
            if it % (self.cfg.save_every_it - 1) == 0:
                cur_train_loss_dict_inr = {k: v / (it + 1) for k, v in train_loss_dict_inr.items()}
                train_msg = self.create_loss_message(cur_train_loss_dict_inr,
                                                     expr_ID=self.cfg.expr_ID,
                                                     epoch_num=self.epochs_completed,
                                                     model_name='MotionInfilling',
                                                     it=it,
                                                     try_num=self.try_num,
                                                     mode='train')

                self.logger(train_msg)

        train_loss_dict_inr = {k: v / len(self.ds_train) for k, v in train_loss_dict_inr.items()}
        return train_loss_dict_inr

    def evaluate(self):
        self.inr.eval()

        eval_loss_dict_inr = {}

        dataset = self.ds_val
        # dataset = self.ds_train

        with torch.no_grad():
            for it, data in enumerate(dataset):
                data = {k: data[k].to(self.device) for k in data.keys()}
                data['motion_img'] = data['motion_img'][:, :338, :64]
                bs, _, width = data['motion_img'].shape

                obj_ori_trans = data['obj_traj_computed'][:, :, 0]
                obj_ori_orien = data['obj_orient'][:, :, 0]
                ori_ho_contact = data['ho_contact'][:, :, 0]
                drec_inr = self.inr(data['obj_traj_computed'], obj_ori_trans, obj_ori_orien, ori_ho_contact, width, 3+6+15, self.device)

                loss_total_inr, cur_loss_dict_inr = self.loss_inr(data, drec_inr)
                eval_loss_dict_inr = {k: eval_loss_dict_inr.get(k, 0.0) + v.item() for k, v in
                                             cur_loss_dict_inr.items()}

            eval_loss_dict_inr = {k: v / len(dataset) for k, v in eval_loss_dict_inr.items()}

        return eval_loss_dict_inr

    def compute_geodesic_loss(self, mat1, mat2):
        bs = mat1.shape[0]
        rotmat1 = CRot2rotmat(mat1)
        rotmat2 = CRot2rotmat(mat2)
        diff = torch.bmm(rotmat1, rotmat2.permute(0, 2, 1))
        batch_trace = diff.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)  # Compute batch trace

        numerator = torch.where(batch_trace>0, batch_trace-(1.+1e-6), batch_trace-(1.-5e-3))
        denominator = 2.
        loss = torch.acos(numerator / denominator).sum() / bs  # Add a small number to the denominator due to numerical instability
        return loss

    def compute_variation_Loss(self, img, weight):
        bs, h, w = img.shape
        # tv_h = torch.pow(img[:, 1:, :]-img[:, :-1, :], 2).sum()
        tv_w = torch.pow(img[:, :, 1:]-img[:, :, :-1], 2).sum()
        # return weight * (tv_h + tv_w) / (bs * h * w)
        return weight * tv_w / (bs * w)

    def loss_inr(self, data, drec):
        # loss_transl_offset = 36 * self.LossL1(100 * data['obj_offset'], drec['imgs'][:, :3, :])
        loss_transl_offset = 100 * self.LossL2(100 * data['obj_offset'], drec['imgs'][:, :3, :])

        loss_orient = 36 * self.LossL1(data['obj_orient'], drec['imgs'][:, 3:9, :])

        loss_ho_contact = 36 * self.LossL1(data['ho_contact'], drec['imgs'][:, 9:, :])

        loss_dict = {
            'loss_transl_offset': loss_transl_offset,
            'loss_orient': loss_orient,
            'loss_ho_contact': loss_ho_contact,
        }

        loss_total = torch.stack(list(loss_dict.values())).sum()
        loss_dict['loss_total'] = loss_total

        return loss_total, loss_dict

    def generate_mask(self, bs, width, height, ratio):
        if self.device == torch.device("cpu"):
            mask = torch.FloatTensor(height, width).uniform_() > ratio
        else:
            mask = torch.cuda.FloatTensor(height, width).uniform_() > ratio

        mask[:, 0] = torch.ones(height)
        mask[:, -1] = torch.ones(height)
        mask = mask.repeat(bs, 1, 1)

        return mask

    def fit(self, n_epochs=None):
        starttime = datetime.now().replace(microsecond=0)
        if n_epochs is None:
            n_epochs = self.cfg.n_epochs

        self.logger(
            'Started Training at %s for %d epochs' % (datetime.strftime(starttime, '%Y-%m-%d_%H:%M:%S'), n_epochs))

        prev_lr_inr = np.inf
        self.fit_inr = True

        lr_scheduler_inr = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_inr, 'min')

        for epoch_num in range(1, n_epochs + 1):
            self.logger('--- starting Epoch # %03d' % epoch_num)

            train_loss_dict_inr = self.train()
            eval_loss_dict_inr = self.evaluate()

            if self.fit_inr:
                lr_scheduler_inr.step(eval_loss_dict_inr['loss_total'])
                cur_lr_inr = self.optimizer_inr.param_groups[0]['lr']

                if cur_lr_inr != prev_lr_inr:
                    self.logger('--- Motion Infilling learning rate changed from %.2e to %.2e ---' % (
                        prev_lr_inr, cur_lr_inr))
                    prev_lr_inr = cur_lr_inr

                with torch.no_grad():
                    eval_msg = Trainer.create_loss_message(eval_loss_dict_inr, expr_ID=self.cfg.expr_ID,
                                                           epoch_num=self.epochs_completed, it=len(self.ds_val),
                                                           model_name='Motion Infilling',
                                                           try_num=self.try_num, mode='evald')
                    if eval_loss_dict_inr['loss_total'] < self.best_loss_inr and epoch_num > 0:
                        self.cfg.best_inr = makepath(os.path.join(self.cfg.work_dir, 'snapshots',
                                                                         'TR%02d_E%04d_inr.pt' % (
                                                                             self.try_num, self.epochs_completed)),
                                                            isfile=True)
                        self.save_inr()
                        self.logger(eval_msg + ' ** ')
                        self.best_loss_inr = eval_loss_dict_inr['loss_total']

                    self.swriter.add_scalars('total_loss_inr/scalars',
                                             {'train_loss_total': train_loss_dict_inr['loss_total'],
                                              'evald_loss_total': eval_loss_dict_inr['loss_total'], },
                                             self.epochs_completed)

            self.epochs_completed += 1

        endtime = datetime.now().replace(microsecond=0)

        self.logger('Finished Training at %s\n' % (datetime.strftime(endtime, '%Y-%m-%d_%H:%M:%S')))
        self.logger(
            'Training done in %s!\n' % (endtime - starttime))
        self.logger('Best Motion Infilling val total loss achieved: %.2e\n' % (self.best_loss_inr))
        self.logger('Best Motion Infilling model path: %s\n' % self.cfg.best_inr)

    def save_inr(self):
        torch.save(self.inr.module.state_dict()
                   if isinstance(self.inr, torch.nn.DataParallel)
                   else self.inr.state_dict(), self.cfg.best_inr)

    @staticmethod
    def create_loss_message(loss_dict, expr_ID='XX', epoch_num=0, model_name='Motion Infilling', it=0, try_num=0,
                            mode='evald'):
        ext_msg = ' | '.join(['%s = %.2e' % (k, v) for k, v in loss_dict.items() if k != 'loss_total'])
        return '[%s]_TR%02d_E%03d - It %05d - %s - %s: [T:%.2e] - [%s]' % (
            expr_ID, try_num, epoch_num, it, model_name, mode, loss_dict['loss_total'], ext_msg)