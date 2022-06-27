from typing import List, Dict, Union, Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from firelab.config import Config

from models.inrs import FourierINRs, SIRENs, HierarchicalFourierINRs
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

        self.frame_D = 330
        self.latent_D = 512
        self.dim_z = 1024
        self.fframe_enc = ResBlock(self.frame_D, self.latent_D)
        self.lframe_enc = ResBlock(self.frame_D, self.latent_D)

        self.width, self.height = 64, self.frame_D
        self.img_enc = ResBlock(self.width * self.height, self.dim_z)
        self.feat_enc = ResBlock(self.latent_D * 2, self.dim_z)
        self.enc_mu = nn.Linear(self.dim_z, self.dim_z)
        self.enc_var = nn.Linear(self.dim_z, self.dim_z)

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

    def forward(self, img: Tensor, first_frame: Tensor, last_frame: Tensor, width: int, height: int) -> Dict[
    # def forward(self, first_frame: Tensor, last_frame: Tensor, width: int, height: int) -> Dict[
        str, Union[Union[Tensor, float], Any]]:
        feat_fframe = self.fframe_enc(first_frame)
        feat_lframe = self.lframe_enc(last_frame)
        bs = img.shape[0]
        feat_img = self.img_enc(img.reshape(bs, self.width * self.height))
        dist = torch.distributions.normal.Normal(self.enc_mu(feat_img), F.softplus(self.enc_var(feat_img)))
        # feat_fl = self.feat_enc(torch.cat([feat_fframe, feat_lframe], dim=1))
        # dist = torch.distributions.normal.Normal(self.enc_mu(feat_fl), F.softplus(self.enc_var(feat_fl)))

        z = dist.rsample()
        feat = torch.cat([z, feat_fframe, feat_lframe], dim=1)
        inrs_weights = self.compute_model_forward(feat)
        # inrs_weights = self.compute_model_forward(z)  # Test without first and last frames

        imgs = self.forward_for_weights(inrs_weights, width, height)
        results = {'mean': dist.mean, 'std': dist.scale, 'imgs': imgs}

        return results

    def decode(self, z: Tensor, first_frame: Tensor, last_frame: Tensor, width: int, height: int) -> Dict[str, Tensor]:
        feat_fframe = self.fframe_enc(first_frame)
        feat_lframe = self.lframe_enc(last_frame)
        feat = torch.cat([z, feat_fframe, feat_lframe], dim=1)
        inrs_weights = self.compute_model_forward(feat)
        # inrs_weights = self.compute_model_forward(z)  # Test without first and last frames

        return {'imgs': self.forward_for_weights(inrs_weights, width, height)}

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