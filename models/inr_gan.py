from typing import List, Dict, Union, Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from firelab.config import Config

from models.inrs import FourierINRs
from models.inrs.inrs import FourierINRs_obj
from models.layers import create_activation
from utils.training_utils import sample_noise


class ResBlock(nn.Module):

    def __init__(self,
                 Fin,
                 Fout,
                 n_neurons=512):
        super(ResBlock, self).__init__()
        # Feature dimension of input and output
        self.Fin = Fin
        self.Fout = Fout

        self.fc1 = nn.Linear(Fin, n_neurons)
        self.bn1 = nn.BatchNorm1d(n_neurons)

        self.fc2 = nn.Linear(n_neurons, Fout)
        self.bn2 = nn.BatchNorm1d(Fout)

        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_nl=True):
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))

        Xout = self.fc1(x)
        Xout = self.bn1(Xout)
        Xout = self.ll(Xout)

        Xout = self.fc2(Xout)
        Xout = self.bn2(Xout)
        Xout = Xin + Xout

        if final_nl:
            return self.ll(Xout)
        return Xout


class INRGenerator(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.mapping_network = None
        self.connector = None
        self.size_sampler = None
        self.class_embedder = None
        self.config = config
        self.inr = FourierINRs(self.config)
        # self.inr = FourierINRs(self.config)
        self.dist = None

        self.frame_D = 338
        self.latent_D = 512
        self.dim_z = 1024
        self.fframe_enc = ResBlock(self.frame_D, self.latent_D)
        self.lframe_enc = ResBlock(self.frame_D, self.latent_D)

        self.width, self.height = 64, self.frame_D
        self.img_enc = ResBlock(self.width * self.height, self.dim_z)
        self.enc_mu = nn.Linear(self.dim_z, self.dim_z)
        self.enc_var = nn.Linear(self.dim_z, self.dim_z)

        self.sigmoid = nn.Sigmoid()

        self.init_model()

    def init_model(self):
        input_dim = self.latent_D * 2 + self.dim_z
        # input_dim = self.dim_z  # Test without first and last frame vectors
        self.class_embedder = nn.Identity()
        self.size_sampler = nn.Identity()

        generator_hid_dim = 2048
        generator_num_layers = 10

        dims = [input_dim] \
               + [generator_hid_dim] * generator_num_layers \
               + [self.inr.num_external_params]

        self.mapping_network = nn.Sequential(
            *[INRGeneratorBlock(dims[i], dims[i + 1], True, is_first_layer=(i == 0)) for i in range(len(dims) - 2)])
        self.connector = nn.Linear(dims[-2], dims[-1])
        # self.connector = ResBlock(dims[-2], dims[-1])

    def forward(self, img: Tensor, first_frame: Tensor, last_frame: Tensor, width: int, height: int, device: str) -> Dict[
    # def forward(self, first_frame: Tensor, last_frame: Tensor, width: int, height: int) -> Dict[
        str, Union[Union[Tensor, float], Any]]:
        feat_fframe = self.fframe_enc(first_frame)
        feat_lframe = self.lframe_enc(last_frame)
        bs = img.shape[0]
        feat_img = self.img_enc(img.reshape(bs, self.width * self.height))
        dist = torch.distributions.normal.Normal(self.enc_mu(feat_img), F.softplus(self.enc_var(feat_img)))
        # feat_fl = self.rb1(torch.cat([feat_fframe, feat_lframe], dim=1))
        # feat_fl = self.rb2(feat_fl)
        # dist = torch.distributions.normal.Normal(self.enc_mu(feat_fl), F.softplus(self.enc_var(feat_fl)))

        '''batch_size = first_frame.shape[0]
        if self.dist is None:
            self.dist = torch.distributions.normal.Normal(
                loc=torch.tensor(np.zeros([batch_size, 1024]), requires_grad=False),
                scale=torch.tensor(np.ones([batch_size, 1024]), requires_grad=False)
            )
        z_s = self.dist.rsample().float().to(device=device)
        feat_content = self.rb1(z_s)
        feat_content = self.rb2(feat_content)
        feat_content = self.rb3(feat_content)
        feat = torch.cat([feat_fframe, feat_content, feat_lframe], dim=1)
        inrs_weights = self.compute_model_forward(feat)'''

        z = dist.rsample()
        feat = torch.cat([feat_fframe, z, feat_lframe], dim=1)
        inrs_weights = self.compute_model_forward(feat)
        # inrs_weights = self.compute_model_forward(z)  # Test without first and last frames

        imgs = torch.zeros(bs, height, width).to(device)
        imgs[:, :, 1:-1] = self.forward_for_weights(inrs_weights, width - 2, height)
        imgs[:, :, 0] = first_frame
        imgs[:, :, -1] = last_frame
        imgs[:, -8:, :] = self.sigmoid(imgs[:, -8:, :])
        results = {'mean': dist.mean, 'std': dist.scale, 'imgs': imgs}
        # results = {'imgs': imgs}

        return results

    def decode(self, z: Tensor, first_frame: Tensor, last_frame: Tensor, width: int, height: int) -> Dict[str, Tensor]:
        feat_fframe = self.fframe_enc(first_frame)
        feat_lframe = self.lframe_enc(last_frame)
        feat = torch.cat([z, feat_fframe, feat_lframe], dim=1)
        inrs_weights = self.compute_model_forward(feat)
        # inrs_weights = self.compute_model_forward(z)  # Test without first and last frames

        # feat_content = self.rb1(z)
        # feat_content = self.rb2(feat_content)
        # feat_content = self.rb3(feat_content)
        # feat = torch.cat([feat_fframe, feat_content, feat_lframe], dim=1)
        # inrs_weights = self.compute_model_forward(feat)

        bs = first_frame.shape[0]
        imgs = torch.zeros(bs, height, width).to(first_frame.device)
        imgs[:, :, 1:-1] = self.forward_for_weights(inrs_weights, width - 2, height)
        imgs[:, :, 0] = first_frame
        imgs[:, :, -1] = last_frame

        return {'imgs': imgs}

    def forward_for_weights(self, inrs_weights: Tensor, width: int, height: int,
                            return_activations: bool = False) -> Tensor:
        generation_result = self.inr.generate_image(
            inrs_weights, width, height, return_activations=return_activations)

        images = generation_result
        return images

    def compute_model_forward(self, z: Tensor) -> Tensor:
        latents = self.mapping_network(z)
        weights = self.connector(latents)

        return weights

    def get_output_matrix_size(self) -> int:
        return self.connector.weight.numel()

    def sample_noise(self, batch_size: int, correction: Config = None) -> Tensor:
        return sample_noise(self.config.hp.generator.dist, self.config.hp.generator.z_dim, batch_size, correction)

    '''
    def generate_image(self, batch_size: int, device: str, width: int, height: int, return_activations: bool = False,
                       return_labels: bool = False) -> Tensor:
        """
        Generates an INR and computes it
        """
        inputs = self.sample_noise(batch_size).to(device)  # [batch_size, z_dim]
        inr_params = self.compute_model_forward(inputs)
        # Generating the images
        generation_result = self.forward_for_weights(
            inr_params, width, height, return_activations=return_activations)
        images = generation_result
        return images
    '''

class INRGenerator_obj(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.mapping_network = None
        self.connector = None
        self.size_sampler = None
        self.class_embedder = None
        self.config = config
        self.inr = FourierINRs_obj(self.config)
        # self.inr = FourierINRs(self.config)
        self.dist = None

        self.d_feat_bps = 256
        self.width = 64
        self.d_feat_finger = 512
        self.d_feat = 64
        self.traj_finger_enc = nn.Sequential(
            ResBlock(15 * self.width, self.d_feat_finger),
            ResBlock(self.d_feat_finger, self.d_feat_finger)
        )
        self.bps_enc = ResBlock(1024, self.d_feat_bps)
        self.obj_ori_trans_enc = ResBlock(3, self.d_feat)
        self.obj_ori_orien_enc = ResBlock(6, self.d_feat)
        self.ho_contact_enc = ResBlock(15, self.d_feat)

        self.init_model()

    def init_model(self):
        input_dim = self.d_feat_finger + self.d_feat_bps + 3 * self.d_feat
        # input_dim = self.dim_z  # Test without first and last frame vectors
        self.class_embedder = nn.Identity()
        self.size_sampler = nn.Identity()

        generator_hid_dim = 1024
        generator_num_layers = 5

        dims = [input_dim] \
               + [generator_hid_dim] * generator_num_layers \
               + [self.inr.num_external_params]

        self.mapping_network = nn.Sequential(
            *[INRGeneratorBlock(dims[i], dims[i + 1], True, is_first_layer=(i == 0)) for i in range(len(dims) - 2)])
        self.connector = nn.Linear(dims[-2], dims[-1])
        # self.connector = ResBlock(dims[-2], dims[-1])

    def forward(self, obj_computed: Tensor, start_bps: Tensor, obj_ori_trans: Tensor, obj_ori_orien: Tensor, ho_contact: Tensor, width: int, height: int, device: str) -> Dict[str, Union[Union[Tensor, float], Any]]:
        bs = obj_computed.shape[0]
        feat_traj_finger = self.traj_finger_enc(obj_computed.reshape(bs, 15*width))
        feat_bps = self.bps_enc(start_bps)
        feat_ori_trans = self.obj_ori_trans_enc(obj_ori_trans)
        feat_ori_orien = self.obj_ori_orien_enc(obj_ori_orien)
        feat_ori_ho_contact = self.ho_contact_enc(ho_contact)

        feat = torch.cat([feat_traj_finger, feat_bps, feat_ori_trans, feat_ori_orien, feat_ori_ho_contact], dim=1)
        inrs_weights = self.compute_model_forward(feat)

        imgs = torch.zeros(obj_computed.shape[0], height, width).to(device)
        imgs[:, :, 1:] = self.forward_for_weights(inrs_weights, width - 1, height)
        imgs[:, :3, 0] = obj_ori_trans
        imgs[:, 3:9, 0] = obj_ori_orien
        imgs[:, 9:, 0] = ho_contact
        results = {'imgs': imgs}

        return results

    def forward_for_weights(self, inrs_weights: Tensor, width: int, height: int,
                            return_activations: bool = False) -> Tensor:
        generation_result = self.inr.generate_image(
            inrs_weights, width, height, return_activations=return_activations)

        images = generation_result
        return images

    def compute_model_forward(self, z: Tensor) -> Tensor:
        latents = self.mapping_network(z)
        weights = self.connector(latents)

        return weights

    def get_output_matrix_size(self) -> int:
        return self.connector.weight.numel()

    def sample_noise(self, batch_size: int, correction: Config = None) -> Tensor:
        return sample_noise(self.config.hp.generator.dist, self.config.hp.generator.z_dim, batch_size, correction)


class INRGeneratorBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, residual: bool, is_first_layer: bool,
                 main_branch_weight: float = 0.1):
        super().__init__()

        layers = [nn.Linear(in_features, out_features)]

        if in_features == out_features and residual and not is_first_layer:
            self.residual = True
            self.main_branch_weight = nn.Parameter(torch.tensor(main_branch_weight))
        else:
            self.residual = False

        layers.append(create_activation('relu', {}))

        self.transform = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        y = self.transform(x)

        if self.residual:
            return x + self.main_branch_weight * y
        else:
            return y


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = 330 * 2
        self.output_dim = 1

        self.fc1 = nn.Linear(self.input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.output_dim)
        self.relu = nn. ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
