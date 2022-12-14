# python3.7
"""Contains the generator class of StyleGAN.

Basically, this class is derived from the `BaseGenerator` class defined in
`base_generator.py`.
"""
import os
import numpy as np

import torch

from . import model_settings
from .base_generator import BaseGenerator
from .stylegan2_pytorch.stylegan2 import models
__all__ = ['StyleGAN2Generator']
import time


class StyleGAN2Generator(BaseGenerator):
    """Defines the generator class of StyleGAN.

    Different from conventional GAN, StyleGAN introduces a disentangled latent
    space (i.e., W space) besides the normal latent space (i.e., Z space). Then,
    the disentangled latent code, w, is fed into each convolutional layer to
    modulate the `style` of the synthesis through AdaIN (Adaptive Instance
    Normalization) layer. Normally, the w's fed into all layers are the same. But,
    they can actually be different to make different layers get different styles.
    Accordingly, an extended space (i.e. W+ space) is used to gather all w's
    together. Taking the official StyleGAN model trained on FF-HQ dataset as an
    instance, there are
    (1) Z space, with dimension (512,)
    (2) W space, with dimension (512,)
    (3) W+ space, with dimension (18, 512)
    """

    def __init__(self, model_name, logger=None,truncation_psi=0.5,randomize_noise=False):
        self.truncation_psi = truncation_psi
        self.truncation_layers = model_settings.STYLEGAN_TRUNCATION_LAYERS
        self.randomize_noise = randomize_noise
        self.model_specific_vars = ['truncation.truncation']
        super().__init__(model_name, logger)
        self.num_layers = (int(np.log2(self.resolution)) - 1) * 2

    def build(self):
        pass

    def load(self):
        self.logger.info(f'Loading pytorch model from `{self.model_path}`.')
        print(f'Loading pytorch model from `{self.model_path}`.')
        self.model = models.load(self.model_path,randomize_noise=self.randomize_noise)
        self.model.eval().to(self.run_device)
        self.model.set_truncation(truncation_psi=self.truncation_psi)


    def sample(self, num, latent_space_type='Z'):
        """Samples latent codes randomly.

        Args:
          num: Number of latent codes to sample. Should be positive.
          latent_space_type: Type of latent space from which to sample latent code.
            Only [`Z`, `W`, `WP`] are supported. Case insensitive. (default: `Z`)

        Returns:
          A `numpy.ndarray` as sampled latend codes.

        Raises:
          ValueError: If the given `latent_space_type` is not supported.
        """
        # np.random.seed(int(time.time()))
        # latent_space_type = latent_space_type.upper()
        # if latent_space_type == 'Z':
        #     latent_codes = np.random.randn(num, self.latent_space_dim)
        # elif latent_space_type == 'W':
        #     latent_codes = np.random.randn(num, self.w_space_dim)
        # elif latent_space_type == 'WP':
        #     latent_codes = np.random.randn(num, self.num_layers, self.w_space_dim)
        # else:
        #     raise ValueError(f'Latent space type `{latent_space_type}` is invalid!')
        #
        # return latent_codes.astype(np.float32)
        rnd = np.random.RandomState(int(time.time()))
        latent_space_type = latent_space_type.upper()
        if latent_space_type == 'Z':
            latent_codes = rnd.randn(num, self.latent_space_dim)
        elif latent_space_type == 'W':
            latent_codes = rnd.randn(num, self.w_space_dim)
        elif latent_space_type == 'WP':
            latent_codes = rnd.randn(num, self.num_layers, self.w_space_dim)
        else:
            raise ValueError(f'Latent space type `{latent_space_type}` is invalid!')

        return latent_codes.astype(np.float32)




    def preprocess(self, latent_codes, latent_space_type='Z'):
        """Preprocesses the input latent code if needed.

        Args:
          latent_codes: The input latent codes for preprocessing.
          latent_space_type: Type of latent space to which the latent codes belong.
            Only [`Z`, `W`, `WP`] are supported. Case insensitive. (default: `Z`)

        Returns:
          The preprocessed latent codes which can be used as final input for the
            generator.

        Raises:
          ValueError: If the given `latent_space_type` is not supported.
        """
        if not isinstance(latent_codes, np.ndarray):
            raise ValueError(f'Latent codes should be with type `numpy.ndarray`!')

        latent_space_type = latent_space_type.upper()
        if latent_space_type == 'Z':
            latent_codes = latent_codes.reshape(-1, self.latent_space_dim)
            norm = np.linalg.norm(latent_codes, axis=1, keepdims=True)
            latent_codes = latent_codes / norm * np.sqrt(self.latent_space_dim)
        elif latent_space_type == 'W':
            latent_codes = latent_codes.reshape(-1, self.w_space_dim)
        elif latent_space_type == 'WP':
            latent_codes = latent_codes.reshape(-1, self.num_layers, self.w_space_dim)
        else:
            raise ValueError(f'Latent space type `{latent_space_type}` is invalid!')

        return latent_codes.astype(np.float32)

    def easy_sample(self, num, latent_space_type='Z'):
        return self.preprocess(self.sample(num, latent_space_type),
                               latent_space_type)

    def synthesize(self,
                   latent_codes,
                   latent_space_type='Z',
                   generate_style=False,
                   generate_image=True):
        """Synthesizes images with given latent codes.

        One can choose whether to generate the layer-wise style codes.

        Args:
          latent_codes: Input latent codes for image synthesis.
          latent_space_type: Type of latent space to which the latent codes belong.
            Only [`Z`, `W`, `WP`] are supported. Case insensitive. (default: `Z`)
          generate_style: Whether to generate the layer-wise style codes. (default:
            False)
          generate_image: Whether to generate the final image synthesis. (default:
            True)

        Returns:
          A dictionary whose values are raw outputs from the generator.
        """
        if not isinstance(latent_codes, np.ndarray):
            raise ValueError(f'Latent codes should be with type `numpy.ndarray`!')

        results = {}

        latent_space_type = latent_space_type.upper()
        latent_codes_shape = latent_codes.shape

        # Generate from Z space.
        if latent_space_type == 'Z':
            if not (len(latent_codes_shape) == 2 and
                    latent_codes_shape[0] <= self.batch_size and
                    latent_codes_shape[1] == self.latent_space_dim):
                raise ValueError(f'Latent_codes should be with shape [batch_size, '
                                 f'latent_space_dim], where `batch_size` no larger '
                                 f'than {self.batch_size}, and `latent_space_dim` '
                                 f'equal to {self.latent_space_dim}!\n'
                                 f'But {latent_codes_shape} received!')
            zs = torch.from_numpy(latent_codes).type(torch.FloatTensor)
            zs = zs.to(self.run_device)
            ws = self.model.run_mapping(latents=zs, labels=None, mapping_grad=True)
            wps = self.model.run_truncation(ws)
            results['z'] = latent_codes
            results['w'] = self.get_value(ws)
            results['wp'] = self.get_value(wps)
        # Generate from W space.
        elif latent_space_type == 'W':
            if not (len(latent_codes_shape) == 2 and
                    latent_codes_shape[0] <= self.batch_size and
                    latent_codes_shape[1] == self.w_space_dim):
                raise ValueError(f'Latent_codes should be with shape [batch_size, '
                                 f'w_space_dim], where `batch_size` no larger than '
                                 f'{self.batch_size}, and `w_space_dim` equal to '
                                 f'{self.w_space_dim}!\n'
                                 f'But {latent_codes_shape} received!')
            ws = torch.from_numpy(latent_codes).type(torch.FloatTensor)
            ws = ws.to(self.run_device)
            wps = self.model.run_truncation(ws)
            results['w'] = latent_codes
            results['wp'] = self.get_value(wps)
        # Generate from W+ space.
        elif latent_space_type == 'WP':
            if not (len(latent_codes_shape) == 3 and
                    latent_codes_shape[0] <= self.batch_size and
                    latent_codes_shape[1] == self.num_layers and
                    latent_codes_shape[2] == self.w_space_dim):
                raise ValueError(f'Latent_codes should be with shape [batch_size, '
                                 f'num_layers, w_space_dim], where `batch_size` no '
                                 f'larger than {self.batch_size}, `num_layers` equal '
                                 f'to {self.num_layers}, and `w_space_dim` equal to '
                                 f'{self.w_space_dim}!\n'
                                 f'But {latent_codes_shape} received!')
            wps = torch.from_numpy(latent_codes).type(torch.FloatTensor)
            wps = wps.to(self.run_device)
            results['wp'] = latent_codes
        else:
            raise ValueError(f'Latent space type `{latent_space_type}` is invalid!')

        if generate_style:
            for i in range(self.num_layers):
                # style = self.model.synthesis.__getattr__(
                style = self.model.__getattr__(
                    f'layer{i}').epilogue.style_mod.dense(wps[:, i, :])
                results[f'style{i:02d}'] = self.get_value(style)

        if generate_image:
            images = self.model.synthesis(wps)
            results['image'] = self.get_value(images)

        return results
    def style_mixing(self,
                   latent_codes,
                     style_range=range(18),
                     style_codes=None,
                     mix_ratio=0.5,
                   latent_space_type='Z',

                   generate_style=False,
                   generate_image=True):
        """Synthesizes images with given latent codes.

        One can choose whether to generate the layer-wise style codes.

        Args:
          latent_codes: Input latent codes for image synthesis.
          latent_space_type: Type of latent space to which the latent codes belong.
            Only [`Z`, `W`, `WP`] are supported. Case insensitive. (default: `Z`)
          generate_style: Whether to generate the layer-wise style codes. (default:
            False)
          generate_image: Whether to generate the final image synthesis. (default:
            True)

        Returns:
          A dictionary whose values are raw outputs from the generator.
        """
        num=latent_codes.shape[0]
        if style_codes.any()==None:
        # if style_codes==None:
            style_codes=self.sample(num,latent_space_type=latent_space_type)

        if not isinstance(latent_codes, np.ndarray):
            raise ValueError(f'Latent codes should be with type `numpy.ndarray`!')

        results = {}

        latent_space_type = latent_space_type.upper()
        latent_codes_shape = latent_codes.shape

        # Generate from Z space.
        if latent_space_type == 'Z':
            if not (len(latent_codes_shape) == 2 and
                    latent_codes_shape[0] <= self.batch_size and
                    latent_codes_shape[1] == self.latent_space_dim):
                raise ValueError(f'Latent_codes should be with shape [batch_size, '
                                 f'latent_space_dim], where `batch_size` no larger '
                                 f'than {self.batch_size}, and `latent_space_dim` '
                                 f'equal to {self.latent_space_dim}!\n'
                                 f'But {latent_codes_shape} received!')
            zs = torch.from_numpy(latent_codes).type(torch.FloatTensor)
            zs = zs.to(self.run_device)
            ws = self.model.run_mapping(latents=zs, labels=None, mapping_grad=True)
            wps = self.model.run_truncation(ws)

            style_zs = torch.from_numpy(style_codes).type(torch.FloatTensor)
            style_zs = style_zs.to(self.run_device)
            style_ws = self.model.run_mapping(latents=style_zs, labels=None, mapping_grad=True)
            style_wps = self.model.run_truncation(style_ws)

            results['z'] = latent_codes
            results['w'] = self.get_value(ws)
            results['wp'] = self.get_value(wps)

            results['style_z'] = style_codes
            results['style_w'] = self.get_value(style_ws)
            results['style_wp'] = self.get_value(style_wps)
        # Generate from W space.
        elif latent_space_type == 'W':
            # print(latent_codes_shape[0])
            # print(latent_codes_shape[1])
            if not (len(latent_codes_shape) == 2 and
                    latent_codes_shape[0] <= self.batch_size and
                    latent_codes_shape[1] == self.w_space_dim):
                raise ValueError(f'Latent_codes should be with shape [batch_size, '
                                 f'w_space_dim], where `batch_size` no larger than '
                                 f'{self.batch_size}, and `w_space_dim` equal to '
                                 f'{self.w_space_dim}!\n'
                                 f'But {latent_codes_shape} received!')
            ws = torch.from_numpy(latent_codes).type(torch.FloatTensor)
            ws = ws.to(self.run_device)
            wps = self.model.run_truncation(ws)

            style_ws =torch.from_numpy(style_codes).type(torch.FloatTensor)
            style_ws = style_ws.to(self.run_device)
            style_wps = self.model.run_truncation(style_ws)

            results['w'] = latent_codes
            results['wp'] = self.get_value(wps)

            results['style_w'] = self.get_value(style_ws)
            results['style_wp'] = self.get_value(style_wps)
        # Generate from W+ space.
        elif latent_space_type == 'WP':
            if not (len(latent_codes_shape) == 3 and
                    latent_codes_shape[0] <= self.batch_size and
                    latent_codes_shape[1] == self.num_layers and
                    latent_codes_shape[2] == self.w_space_dim):
                raise ValueError(f'Latent_codes should be with shape [batch_size, '
                                 f'num_layers, w_space_dim], where `batch_size` no '
                                 f'larger than {self.batch_size}, `num_layers` equal '
                                 f'to {self.num_layers}, and `w_space_dim` equal to '
                                 f'{self.w_space_dim}!\n'
                                 f'But {latent_codes_shape} received!')
            wps = torch.from_numpy(latent_codes).type(torch.FloatTensor)
            wps = wps.to(self.run_device)

            style_wps = torch.from_numpy(style_codes).type(torch.FloatTensor)
            style_wps = style_wps.to(self.run_device)
            results['wp'] = latent_codes
            results['style_wps'] = style_wps
        else:
            raise ValueError(f'Latent space type `{latent_space_type}` is invalid!')

        if generate_style:
            for i in range(self.num_layers):
                # style = self.model.synthesis.__getattr__(
                # style = self.model.__getattr__(
                #     f'layer{i}').epilogue.style_mod.dense(wps[:, i, :])
                style = self.model.__getattr__(
                    f'layer{i}').dense(wps[:, i, :])
                results[f'style{i:02d}'] = self.get_value(style)

        images = self.model.synthesis(wps)
        results['origin_image'] = self.get_value(images)

        mixed_wps=wps

        mixed_wps[:,style_range,:]*=1-mix_ratio
        mixed_wps[:,style_range,:]+=style_wps[:,style_range,:]*mix_ratio

        results['mixed_wps'] = mixed_wps
        if generate_image:
            images = self.model.synthesis(wps)
            results['image'] = self.get_value(images)

        return results

    def easy_style_mixing(self, latent_codes,style_range,style_codes,mix_ratio, **kwargs):
        """Wraps functions `synthesize()` and `postprocess()` together."""
        outputs = self.style_mixing(latent_codes,style_range,style_codes,mix_ratio, **kwargs)
        if 'image' in outputs:
            outputs['image'] = self.postprocess(outputs['image'])
        if 'origin_image' in outputs:
            outputs['origin_image'] = self.postprocess(outputs['origin_image'])

        return outputs

    def dlatent_converter(self,
                          zs,
                          latent_space_type='Z'):
        latent_space_type = latent_space_type.upper()
        latent_codes_shape = zs.shape
        if not (len(latent_codes_shape) == 2 and
                latent_codes_shape[0] <= self.batch_size and
                latent_codes_shape[1] == self.latent_space_dim):
            raise ValueError(f'Latent_codes should be with shape [batch_size, '
                             f'latent_space_dim], where `batch_size` no larger '
                             f'than {self.batch_size}, and `latent_space_dim` '
                             f'equal to {self.latent_space_dim}!\n'
                             f'But {latent_codes_shape} received!')
        if latent_space_type == 'Z':
            return zs
        elif latent_space_type == 'W':
            zs=torch.from_numpy(zs).type(torch.FloatTensor)
            zs=zs.to(self.run_device)
            ws = self.model.run_mapping(latents=zs, labels=None, mapping_grad=True)
            ws=self.get_value(ws)
            return ws
        elif latent_space_type == 'WP':
            zs = torch.from_numpy(zs).type(torch.FloatTensor)
            zs = zs.to(self.run_device)
            ws = self.model.run_mapping(latents=zs, labels=None, mapping_grad=True)
            wps = self.model.run_truncation(ws)
            wps=self.get_value(wps)
            return wps

    def dlatent_processor(self,
                          latent_codes,
                          latent_space_type='Z'):

        latent_space_type = latent_space_type.upper()
        if not isinstance(latent_codes,torch.Tensor):
            latent_codes=torch.from_numpy(latent_codes).type(torch.FloatTensor)
            #latent_codes=latent_codes.to(self.run_device)

        latent_codes_shape = list(latent_codes.size())

        # Generate from Z space.
        if latent_space_type == 'Z':
            if not (len(latent_codes_shape) == 2 and
                    latent_codes_shape[0] <= self.batch_size and
                    latent_codes_shape[1] == self.latent_space_dim):
                raise ValueError(f'Latent_codes should be with shape [batch_size, '
                                 f'latent_space_dim], where `batch_size` no larger '
                                 f'than {self.batch_size}, and `latent_space_dim` '
                                 f'equal to {self.latent_space_dim}!\n'
                                 f'But {latent_codes_shape} received!')
            zs = latent_codes
            zs = zs.to(self.run_device)
            ws = self.model.run_mapping(latents=zs, labels=None, mapping_grad=True)
            wps = self.model.run_truncation(ws)
        # Generate from W space.
        elif latent_space_type == 'W':
            if not (len(latent_codes_shape) == 2 and
                    latent_codes_shape[0] <= self.batch_size and
                    latent_codes_shape[1] == self.w_space_dim):
                raise ValueError(f'Latent_codes should be with shape [batch_size, '
                                 f'w_space_dim], where `batch_size` no larger than '
                                 f'{self.batch_size}, and `w_space_dim` equal to '
                                 f'{self.w_space_dim}!\n'
                                 f'But {latent_codes_shape} received!')
            ws = latent_codes
            ws = ws.to(self.run_device)
            wps = self.model.run_truncation(ws)
        # Generate from W+ space.
        elif latent_space_type == 'WP':
            if not (len(latent_codes_shape) == 3 and
                    latent_codes_shape[0] <= self.batch_size and
                    latent_codes_shape[1] == self.num_layers and
                    latent_codes_shape[2] == self.w_space_dim):
                raise ValueError(f'Latent_codes should be with shape [batch_size, '
                                 f'num_layers, w_space_dim], where `batch_size` no '
                                 f'larger than {self.batch_size}, `num_layers` equal '
                                 f'to {self.num_layers}, and `w_space_dim` equal to '
                                 f'{self.w_space_dim}!\n'
                                 f'But {latent_codes_shape} received!')
            wps =latent_codes
            wps = wps.to(self.run_device)
        else:
            raise ValueError(f'Latent space type `{latent_space_type}` is invalid!')

        return wps