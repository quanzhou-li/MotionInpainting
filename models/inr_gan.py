from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from firelab.config import Config

from models.inrs import FourierINRs
from models.layers import create_activation
from utils.training_utils import sample_noise


class INRGenerator(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.mapping_network = None
        self.connector = None
        self.size_sampler = None
        self.class_embedder = None
        self.config = config
        self.inr = FourierINRs(self.config)
        self.init_model()

    def init_model(self):
        input_dim = 512
        self.class_embedder = nn.Identity()
        self.size_sampler = nn.Identity()

        generator_hid_dim = 1024
        generator_num_layers = 3

        dims = [input_dim] \
               + [generator_hid_dim] * generator_num_layers \
               + [self.inr.num_external_params]

        self.mapping_network = nn.Sequential(
            *[INRGeneratorBlock(dims[i], dims[i + 1], True, is_first_layer=(i == 0)) for i in range(len(dims) - 2)])
        self.connector = nn.Linear(dims[-2], dims[-1])

        self.connector.bias.data.mul_(np.sqrt(1 / dims[1]))

    def forward(self, z: Tensor, img_size: int = None) -> Tensor:
        img_size = 256 if img_size is None else img_size
        inrs_weights = self.compute_model_forward(z)

        return self.forward_for_weights(inrs_weights, img_size)

    def forward_for_weights(self, inrs_weights: Tensor, img_size: int = None,
                            return_activations: bool = False) -> Tensor:
        img_size = 256 if img_size is None else img_size
        generation_result = self.inr.generate_image(
            inrs_weights, img_size, return_activations=return_activations)

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

    def generate_image(self, batch_size: int, device: str, return_activations: bool = False,
                       return_labels: bool = False) -> Tensor:
        """
        Generates an INR and computes it
        """
        inputs = self.sample_noise(batch_size).to(device)  # [batch_size, z_dim]
        aspect_ratios = None  # In case of variable-sized generation
        inr_params = self.compute_model_forward(inputs)

        # Generating the images
        generation_result = self.forward_for_weights(
            inr_params, aspect_ratios=aspect_ratios, return_activations=return_activations)

        images = generation_result
        return images


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
