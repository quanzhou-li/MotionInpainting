import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from firelab.config import Config

from models.layers import create_activation
from models.inrs.modules import (
    INRProxy,
    INRLinear,
    INRSELinear,
    INRSharedLinear,
    INRmFiLM,
    INRAdaIN,
    INRFactorizedLinear,
    INRFactorizedSELinear,
    INRResidual,
    INRSequential,
    INRInputSkip,
    MultiModalINRFactorizedSELinear,
    MultiModalINRSharedLinear,
    INRSVDLinear,
)


def generate_coords(batch_size: int, width: int, height: int) -> Tensor:
    row = torch.arange(0, height).float() / height
    col = torch.arange(0, width).float() / width
    x_coords = row.view(1, -1).repeat(width, 1)
    y_coords = col.view(1, -1).repeat(height, 1)
    y_coords = y_coords.t()

    coords = torch.stack([x_coords, y_coords], dim=2) # [width, height, 2]
    coords = coords.view(-1, 2) # [width * height, 2]
    coords = coords.t().view(1, 2, width * height).repeat(batch_size, 1, 1) # [batch_size, 2, n_coords]

    return coords

class INRs(nn.Module):
    def __init__(self, config: Config):
        super(INRs, self).__init__()

        self.config = config
        self.init_model()
        self.num_external_params = sum(m.num_external_params for m in self.model.children())
        self.num_shared_params = sum(p.numel() for p in self.parameters())
        self.min_scale = 1.0 # In some setting, it is going to be changed over time

    def init_model(self):
        raise NotImplementedError("It is a base class. Implement in `.init_model()` in your child class.")

    @torch.no_grad()
    def generate_input_coords(self, batch_size: int, width: int, height: int) -> Tensor:
        """
        @param batch_size
        @return coords # [batch_size, coord_dim, n_coords]
        """
        coords = generate_coords(batch_size, width, height)
        return coords

    def generate_image(self, inrs_weights: Tensor, width: int, height: int, return_activations: bool=False) -> Tensor:
        coords = self.generate_input_coords(len(inrs_weights), width, height).to(inrs_weights.device)

        if return_activations:
            images_raw, activations = self.forward(coords, inrs_weights, return_activations=True) # [batch_size, num_channels, num_coords]
        else:
            images_raw = self.forward(coords, inrs_weights) # [batch_size, num_channels, num_coords]

        num_img_channels = 1
        images = images_raw.view(len(inrs_weights), num_img_channels, width, height) # [batch_size, num_channels, img_size, img_size]

        return (images, activations) if return_activations else images

    def apply_weights(self, x: Tensor, inrs_weights: Tensor, return_activations: bool=False) -> Tensor:
        curr_w = inrs_weights

        if return_activations:
            activations = {'initial': x.cpu().detach()}

        for i, module in enumerate(self.model.children()):
            module_params = curr_w[:, :module.num_external_params]
            x = module(x, module_params)
            curr_w = curr_w[:, module.num_external_params:]

            if return_activations:
                activations[f'{i}-{module}'] = x.cpu().detach()

        assert curr_w.numel() == 0, f"Not all params were used: {curr_w.shape}"

        return (x, activations) if return_activations else x

    def forward(self, coords: Tensor, inrs_weights: Tensor, return_activations: bool=False) -> Tensor:
        """
        Computes a batch of INRs in the given coordinates

        @param coords: coordinates | [n_coords, 2]
        @param inrs_weights: weights of INRs | [batch_size, coord_dim]
        """
        return self.apply_weights(coords, inrs_weights, return_activations=return_activations)

    def create_transform(self, in_features: int, out_features: int, layer_type: str='linear',
                         is_coord_layer: bool=False, weight_std: float=None, bias_std: float=None):
        TYPE_TO_INR_CLASS = {
            'linear': INRLinear,
            'se_linear': INRSELinear,
            'shared_linear': INRSharedLinear,
            'adain_linear': INRSharedLinear,
            'mfilm': INRmFiLM,
            'factorized': INRFactorizedLinear,
            'se_factorized': INRFactorizedSELinear,
            'mm_se_factorized': MultiModalINRFactorizedSELinear,
            'mm_shared_linear': MultiModalINRSharedLinear,
            'svd_linear': INRSVDLinear,
        }

        weight_std = self.compute_weight_std(in_features, is_coord_layer) if weight_std is None else weight_std
        bias_std = self.compute_bias_std(in_features, is_coord_layer) if bias_std is None else bias_std
        layers = [TYPE_TO_INR_CLASS[layer_type](
            in_features,
            out_features,
            weight_std=weight_std,
            bias_std=bias_std,
            **self.config.hp.inr.get(f'module_kwargs.{layer_type}', {}))]

        if layer_type == 'adain_linear':
            layers.append(INRAdaIN(out_features))

        return layers

    def compute_weight_std(self, in_features: int, is_coord_layer: bool) -> float:
        raise NotImplementedError

    def compute_bias_std(self, in_features: int, is_coord_layer: bool) -> float:
        additional_scale = 0.2
        return self.compute_weight_std(in_features, is_coord_layer) * additional_scale


class FourierINRs(INRs):
    def __init__(self, config: Config):
        super().__init__(config)

    def init_model(self):
        layer_sizes = [256, 256, 256]
        layers = self.create_transform(
            2,
            layer_sizes[0],
            layer_type='linear',
            is_coord_layer=True)
        layers.append(INRProxy(create_activation('sine')))

        hid_layers = []

        for i in range(len(layer_sizes) - 1):
            input_dim = layer_sizes[i]

            curr_transform_layers = self.create_transform(
                input_dim,
                layer_sizes[i+1],
                layer_type='se_factorized')
            curr_transform_layers.append(INRProxy(create_activation('relu')))

            hid_layers.append(INRResidual(INRSequential(*curr_transform_layers)))

        layers.append(INRInputSkip(*hid_layers))

        layers.extend(self.create_transform(layer_sizes[-1], 1, 'linear'))
        layers.append(INRProxy(create_activation('none')))

        self.model = nn.Sequential(*layers)

    def compute_weight_std(self, in_features: int, is_coord_layer: bool) -> float:
        if is_coord_layer:
            return 10.0
        else:
            return np.sqrt(2 / in_features)

