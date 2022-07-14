from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from firelab.config import Config

from models.layers import create_activation, Sine
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
    INRFourierFeats,
    INRResConnector,
    INRIdentity,
    INRToRGB,
    INRCoordsSkip,
    INRModuleDict,
    INRModule,
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
        # images = images_raw.view(len(inrs_weights), num_img_channels, width, height).permute(0, 1, 3, 2) # [batch_size, num_channels, img_size, img_size]
        images = images_raw.view(len(inrs_weights), num_img_channels, height, width) # [batch_size, num_channels, img_size, img_size]

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


class SIRENs(INRs):
    def __init__(self, config: Config):
        super().__init__(config)

    def init_model(self):
        layer_sizes = [128, 128, 128]
        layers = self.create_transform(
            2,
            layer_sizes[0],
            'linear', # First layer is of full control
            is_coord_layer=True)
        layers.append(INRProxy(Sine(scale=1.0)))

        for i in range(len(layer_sizes) - 1):
            layers.extend(self.create_transform(
                layer_sizes[i],
                layer_sizes[i+1],
                layer_type='se_factorized')) # Middle layers are large so they are controlled via AdaIN
            layers.append(INRProxy(Sine(scale=1.0)))

        layers.extend(self.create_transform(
            layer_sizes[-1],
            1,
            layer_type='linear' # The last layer is small so let's also control it fully
        ))
        layers.append(INRProxy(create_activation('none')))

        self.model = nn.Sequential(*layers)

    def compute_weight_std(self, in_features: int, is_coord_layer: bool) -> float:
        weight_std = np.sqrt(2 / in_features)
        return weight_std


class FourierINRs(INRs):
    def __init__(self, config: Config):
        self.num_fourier_feats = 5
        super().__init__(config)

    def init_model(self):
        # layer_sizes = [128, 256, 512, 512, 512, 256, 128]
        layer_sizes = [128, 256, 256, 128]
        layers = self.create_transform(
            # self.num_fourier_feats * 2,
            2,
            layer_sizes[0],
            layer_type='linear',
            is_coord_layer=True,
            # weight_std=1.0,
            # bias_std=1.0
        )
        layers.append(INRProxy(create_activation('relu')))


        hid_layers = []

        for i in range(len(layer_sizes) - 1):
            input_dim = layer_sizes[i]

            curr_transform_layers = self.create_transform(
                input_dim,
                layer_sizes[i+1],
                layer_type='mm_se_factorized')
                # layer_type='se_factorized')
            curr_transform_layers.append(INRProxy(create_activation('relu')))

            hid_layers.append((INRSequential(*curr_transform_layers)))

        layers.append(INRInputSkip(*hid_layers))

        '''
        for i in range(len(layer_sizes)-1):
            layers.extend(self.create_transform(layer_sizes[i], layer_sizes[i+1], 'linear'))
            layers.append(INRProxy(create_activation('sine')))
        '''

        layers.extend(self.create_transform(layer_sizes[-1], 1, 'linear'))
        layers.append(INRProxy(create_activation('none')))

        self.model = nn.Sequential(*layers)

        basis_matrix = torch.randn(self.num_fourier_feats, 2)
        self.basis_matrix = nn.Parameter(basis_matrix, requires_grad=False)

    def compute_fourier_feats(self, coords: Tensor) -> Tensor:
        bs = coords.shape[0]
        bm = self.basis_matrix.repeat(bs, 1).reshape(bs, self.num_fourier_feats, 2)
        sines = (2 * np.pi * torch.bmm(bm, coords)).sin()
        cosines = (2 * np.pi * torch.bmm(bm, coords)).sin()
        return torch.cat([sines, cosines], dim=1)

    def compute_weight_std(self, in_features: int, is_coord_layer: bool) -> float:
        if is_coord_layer:
            return 10.0
        else:
            return np.sqrt(2 / in_features)

    # def forward(self, coords: Tensor, inrs_weights: Tensor, return_activations: bool=False) -> Tensor:
        """
        Computes a batch of INRs in the given coordinates

        @param coords: coordinates | [n_coords, 2]
        @param inrs_weights: weights of INRs | [batch_size, coord_dim]
        """
        # return self.apply_weights(self.compute_fourier_feats(coords), inrs_weights, return_activations=return_activations)


class HierarchicalFourierINRs(FourierINRs, INRs):
    """
    Hierarchical INRs operate in a bit different regime:
    we first generate 8x8 resolution, then 16x16, etc...
    This allows us to use larger layer sizes at the beginning
    """
    def __init__(self, config: Config):
        nn.Module.__init__(self)

        self.config = config
        self.init_model()
        self.num_external_params = self.model.num_external_params
        self.num_shared_params = sum(p.numel() for p in self.parameters())

    def init_model(self):
        blocks = []
        resolutions = self.generate_img_sizes(self.config.data.target_img_size)
        res_configs = [self.config.hp.inr.resolutions_params[resolutions[i]] for i in range(self.config.hp.inr.num_blocks)]

        # print('resolution:', [c.resolution for c in res_configs])
        # print('dim:', [c.dim for c in res_configs])
        # print('num_learnable_coord_feats:', [c.num_learnable_coord_feats for c in res_configs])
        # print('to_rgb:', [c.to_rgb for c in res_configs])
        num_to_rgb_blocks = sum(c.to_rgb for c in res_configs)

        for i, res_config in enumerate(res_configs):
            # 1. Creating coord fourier feat embedders for each resolution
            coord_embedder = INRSequential(
                INRFourierFeats(res_config),
                INRProxy(create_activation('sines_cosines')))
            coord_feat_dim = coord_embedder[0].get_num_feats() * 2

            # 2. Main branch. First need does not need any wiring, but later layers use it.
            # A good thing is that we do not need skip-coords anymore.
            if i > 0:
                # Different-resolution blocks are wired together with the connector
                connector_layers = [INRResConnector(
                    res_configs[i - 1].dim,
                    coord_feat_dim,
                    res_config.dim,
                    'nearest',
                    **{"rank": 10, "equalized_lr": False}),
                    INRProxy(create_activation('relu', {}))]
                connector = INRSequential(*connector_layers)
            else:
                connector = INRIdentity()

            transform_layers = []
            for j in range(res_config.n_layers):
                if i == 0 and j == 0:
                    input_size = coord_feat_dim # Since we do not have previous feat dims
                elif self.config.hp.inr.skip_coords:
                    input_size = coord_feat_dim + res_config.dim
                else:
                    input_size = res_config.dim

                transform_layers.extend(self.create_transform(
                    input_size, res_config.dim,
                    layer_type='se_factorized'))

                transform_layers.append(INRProxy(create_activation('relu', {})))

            if res_config.to_rgb or i == (self.config.hp.inr.num_blocks - 1):
                to_rgb_weight_std = self.compute_weight_std(res_config.dim, is_coord_layer=False)
                to_rgb_bias_std = self.compute_bias_std(res_config.dim, is_coord_layer=False)

                to_rgb = INRToRGB(
                    res_config.dim,
                    'none',
                    'nearest',
                    to_rgb_weight_std,
                    to_rgb_bias_std)
            else:
                to_rgb = INRIdentity()

            if self.config.hp.inr.skip_coords:
                transform = INRCoordsSkip(*transform_layers, concat_to_the_first=i > 0)
            else:
                transform = INRSequential(*transform_layers)

            blocks.append(INRModuleDict({
                'coord_embedder': coord_embedder,
                'transform': transform,
                'connector': connector,
                'to_rgb': to_rgb,
            }))

        self.model = INRModuleDict({f'b_{i}': b for i, b in enumerate(blocks)})

    def create_res_config(self, block_idx: int) -> Config:
        increase_conf = self.config.hp.inr.res_increase_scheme
        num_blocks = self.config.hp.inr.num_blocks
        resolutions = self.generate_img_sizes(self.config.data.target_img_size)
        fourier_scale = np.linspace(increase_conf.fourier_scales.min, increase_conf.fourier_scales.max, num_blocks)[block_idx]
        dim = np.linspace(increase_conf.dims.max, increase_conf.dims.min, num_blocks).astype(int)[block_idx]
        num_coord_feats = np.linspace(increase_conf.num_coord_feats.max, increase_conf.num_coord_feats.min, num_blocks).astype(int)[block_idx]

        return Config({
            'resolution': resolutions[block_idx],
            'num_learnable_coord_feats': num_coord_feats.item(),
            'use_diag_feats': resolutions[block_idx] <= increase_conf.diag_feats_threshold,
            'max_num_fixed_coord_feats': 10000 if increase_conf.use_fixed_coord_feats else 0,
            'dim': dim.item(),
            'fourier_scale': fourier_scale.item(),
            'to_rgb': resolutions[block_idx] >= increase_conf.to_rgb_res_threshold,
            'n_layers': 1
        })

    def generate_image(self, inrs_weights: Tensor, img_size: int, aspect_ratios=None, return_activations: bool=False) -> Tensor:
        # Generating coords for each resolution
        batch_size = len(inrs_weights)
        img_sizes = self.generate_img_sizes(img_size)

        coords_list = [generate_coords(batch_size, s, s) for s in img_sizes] # (num_blocks, [batch_size, 2, num_coords_in_block])

        if return_activations:
            images_raw, activations = self.forward(coords_list, inrs_weights, return_activations=True) # [batch_size, num_channels, num_coords]
        else:
            images_raw = self.forward(coords_list, inrs_weights) # [batch_size, num_channels, num_coords]

        images = images_raw.view(batch_size, 3, img_size, img_size) # [batch_size, num_channels, img_size, img_size]

        return (images, activations) if return_activations else images

    def apply_weights(self,
                      coords_list: List[Tensor],
                      inrs_weights: Tensor,
                      return_activations: bool=False,
                      noise_injections: List[Tensor]=None) -> Tensor:

        device = inrs_weights.device
        curr_w = inrs_weights
        images = None

        if return_activations:
            activations = {}

        for i in range(self.config.hp.inr.num_blocks):
            coords = coords_list[i].to(device)
            block = self.model[f'b_{i}']
            curr_w, coord_feats = self.apply_module(curr_w, block.coord_embedder, coords)

            if i == 0:
                if self.config.hp.inr.skip_coords:
                    curr_w, x = self.apply_module(curr_w, block.transform, coord_feats, coord_feats)
                else:
                    curr_w, x = self.apply_module(curr_w, block.transform, coord_feats)
            else:
                # Apply a connector
                curr_w, x = self.apply_module(curr_w, block.connector[0], x, coord_feats) # transform
                if return_activations:
                    activations[f'block-{i}-connector'] = x.cpu().detach()
                curr_w, x = self.apply_module(curr_w, block.connector[1], x) # activation

                # Apply a transform
                if self.config.hp.inr.skip_coords:
                    curr_w, x = self.apply_module(curr_w, block.transform, x, coord_feats)
                else:
                    curr_w, x = self.apply_module(curr_w, block.transform, x)

            if return_activations:
                activations[f'block-{i}'] = x.cpu().detach()

            if isinstance(block.to_rgb, INRToRGB):
                # Converting to an image (possibly using the skip)
                curr_w, images = self.apply_module(curr_w, block.to_rgb, x, images)

                if return_activations:
                    activations[f'block-{i}-images'] = images.cpu().detach()

        if self.config.hp.inr.output_activation == 'tanh':
            out = images.tanh()
        elif self.config.hp.inr.output_activation == 'clamp':
            out = images.clamp(-1.0, 1.0)
        elif self.config.hp.inr.output_activation == 'sigmoid':
            out = images.sigmoid() * 2 - 1
        elif self.config.hp.inr.output_activation in ['none', None]:
            out = images
        else:
            raise NotImplementedError(f'Unknown output activation: {self.config.hp.inr.output_activation}')

        if return_activations:
            activations[f'block-final'] = out.cpu().detach()

        assert curr_w.numel() == 0, f"Not all params were used: {curr_w.shape}"

        return (out, activations) if return_activations else out

    def apply_module(self, curr_w: Tensor, module: INRModule, *inputs) -> Tuple[Tensor, Tensor]:
        """
        Applies params and returns the remaining ones
        """
        module_params = curr_w[:, :module.num_external_params]
        y = module(*inputs, module_params)
        remaining_w = curr_w[:, module.num_external_params:]

        return remaining_w, y

    def generate_img_sizes(self, target_img_size: int) -> List[int]:
        """
        Generates coord features for each resolution to produce a final image
        The main logic is in computing resolutions
        """
        if self.config.hp.inr.res_increase_scheme.enabled:
            return self.generate_linear_img_sizes(target_img_size)
        else:
            return self.generate_exp_img_sizes(target_img_size)

    def generate_exp_img_sizes(self, target_img_size: int) -> List[int]:
        # This determines an additional upscale factor for the upscaling block
        extra_upscale_factor = target_img_size // self.config.data.target_img_size
        img_sizes = []
        curr_img_size = target_img_size

        for i in range(self.config.hp.inr.num_blocks - 1, -1, -1):
            img_sizes.append(curr_img_size)

            if i == self.config.hp.inr.upsample_block_idx:
                curr_img_size = curr_img_size // (extra_upscale_factor * 2)
            else:
                curr_img_size = curr_img_size // 2

        return list(reversed(img_sizes))

    def generate_linear_img_sizes(self, target_img_size: int) -> List[int]:
        min_size = self.config.hp.inr.res_increase_scheme.min_resolution
        max_size = target_img_size
        img_sizes = np.linspace(min_size, max_size, self.config.hp.inr.num_blocks).astype(int)

        return img_sizes.tolist()

    def forward(self, coords_list: List[Tensor], inrs_weights: Tensor, return_activations: bool=False) -> Tensor:
        """
        Computes a batch of INRs in the given coordinates
        @param coords_list: coordinates | (num_blocks, [batch_size, n_coords, 2])
        @param inrs_weights: weights of INRs | [batch_size, coord_dim]
        """
        return self.apply_weights(coords_list, inrs_weights, return_activations=return_activations)