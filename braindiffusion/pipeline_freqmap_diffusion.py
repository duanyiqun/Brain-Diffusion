# This is brain waves to spectron with or without mel
# Migrated from https://github.com/huggingface/diffusers and AudioDiffusion
# Under liscence
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from math import acos, sin
from typing import List, Tuple, Union

import numpy as np
import torch
from diffusers import (
    AudioPipelineOutput,
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    DiffusionPipeline,
    ImagePipelineOutput,
    UNet2DConditionModel,
)
from diffusers.utils import BaseOutput
from PIL import Image

from braindiffusion.utils.wave_spectron import Spectro, Spectro_STFT
from braindiffusion.utils.mne_visualize import visualize_eeg1020, visualize_eeg128, visualize_feature_map, visualize_feature_map_withtoken, create_topo_heat_maps, stack_topomaps, normalize_feature_map
from braindiffusion.utils.sampling_func import *


class WaveDiffusionPipeline(DiffusionPipeline):
    """
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        vqae ([`AutoencoderKL`]): Variational AutoEncoder for Latent wave Diffusion or None
        unet ([`UNet2DConditionModel`]): UNET model
        Spectro ([`Spectro`]): transform wave <-> spectrogram
        scheduler ([`DDIMScheduler` or `DDPMScheduler`]): de-noising scheduler
    """

    _optional_components = ["vqvae"]

    def __init__(
        self,
        vqvae: AutoencoderKL,
        unet: UNet2DConditionModel,
        Spectro: Spectro,
        scheduler: Union[DDIMScheduler, DDPMScheduler],
    ):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler, Spectro=Spectro, vqvae=vqvae)

    def get_input_dims(self) -> Tuple:
        """Returns dimension of input image

        Returns:
            `Tuple`: (height, width)
        """
        input_module = self.vqvae if self.vqvae is not None else self.unet
        # For backwards compatibility
        sample_size = (
            (input_module.sample_size, input_module.sample_size)
            if type(input_module.sample_size) == int
            else input_module.sample_size
        )
        return sample_size

    def get_default_steps(self) -> int:
        """Returns default number of steps recommended for inference

        Returns:
            `int`: number of steps
        """
        return 50 if isinstance(self.scheduler, DDIMScheduler) else 1000

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        wave_file: str = None,
        raw_wave: np.ndarray = None,
        slice: int = 0,
        start_step: int = 0,
        steps: int = None,
        generator: torch.Generator = None,
        mask_start_secs: float = 0,
        mask_end_secs: float = 0,
        step_generator: torch.Generator = None,
        eta: float = 0,
        noise: torch.Tensor = None,
        encoding: torch.Tensor = None,
        target_ids: list = None,
        return_dict=True,
    ) -> Union[
        Union[AudioPipelineOutput, ImagePipelineOutput],
        Tuple[List[Image.Image], Tuple[int, List[np.ndarray]]],
    ]:
        """Generate random Spectro spectrogram from wave input and convert to wave.

        Args:
            batch_size (`int`): number of samples to generate
            wave_file (`str`): must be a file on disk due to Librosa limitation or
            raw_wave (`np.ndarray`): Wave as numpy array
            slice (`int`): slice number of wave to convert
            start_step (int): step to start from
            steps (`int`): number of de-noising steps (defaults to 50 for DDIM, 1000 for DDPM)
            generator (`torch.Generator`): random number generator or None
            mask_start_secs (`float`): number of seconds of wave to mask (not generate) at start
            mask_end_secs (`float`): number of seconds of wave to mask (not generate) at end
            step_generator (`torch.Generator`): random number generator used to de-noise or None
            eta (`float`): parameter between 0 and 1 used with DDIM scheduler
            noise (`torch.Tensor`): noise tensor of shape (batch_size, 1, height, width) or None
            encoding (`torch.Tensor`): for UNet2DConditionModel shape (batch_size, seq_length, cross_attention_dim)
            return_dict (`bool`): if True return wavePipelineOutput, ImagePipelineOutput else Tuple

        Returns:
            `List[PIL Image]`: Spectro spectrograms (`float`, `List[np.ndarray]`): sample rate and raw waves
        """

        steps = steps or self.get_default_steps()
        self.scheduler.set_timesteps(steps)
        step_generator = step_generator or generator
        # For backwards compatibility
        if type(self.unet.sample_size) == int:
            self.unet.sample_size = (self.unet.sample_size, self.unet.sample_size)
        input_dims = self.get_input_dims()
        # print("input_dims", input_dims)
        # sself.Spectro.set_resolution(x_res=input_dims[1], y_res=input_dims[2])
        if noise is None:
            noise = torch.randn(
                (
                    batch_size,
                    self.unet.in_channels,
                    self.unet.sample_size[1],
                    self.unet.sample_size[2],
                ),
                generator=generator,
                device=self.device,
            )
        images = noise
        mask = None

        if wave_file is not None or raw_wave is not None:
            input_image = raw_wave
            input_image = np.frombuffer(input_image.tobytes(), dtype="uint8").reshape(
                (input_image.height, input_image.width)
            )
            input_image = (input_image / 255) * 2 - 1
            input_images = torch.tensor(input_image[np.newaxis, :, :], dtype=torch.float).to(self.device)

            if self.vqvae is not None:
                input_images = self.vqvae.encode(torch.unsqueeze(input_images, 0)).latent_dist.sample(
                    generator=generator
                )[0]
                input_images = 0.18215 * input_images

            if start_step > 0:
                images[0, 0] = self.scheduler.add_noise(input_images, noise, self.scheduler.timesteps[start_step - 1])

            pixels_per_second = (
                self.unet.sample_size[1] * self.Spectro.get_sample_rate() / self.Spectro.x_res / self.Spectro.hop_length
            )
            # mask_start = int(mask_start_secs * pixels_per_second)
            # mask_end = int(mask_end_secs * pixels_per_second)
            mask = self.scheduler.add_noise(input_images, noise, torch.tensor(self.scheduler.timesteps[start_step:]))

        for step, t in enumerate(self.progress_bar(self.scheduler.timesteps[start_step:])):
            if isinstance(self.unet, UNet2DConditionModel):
                model_output = self.unet(images, t, encoding)["sample"]
            else:
                model_output = self.unet(images, t)["sample"]

            if isinstance(self.scheduler, DDIMScheduler):
                images = self.scheduler.step(
                    model_output=model_output,
                    timestep=t,
                    sample=images,
                    eta=eta,
                    generator=step_generator,
                )["prev_sample"]
            else:
                images = self.scheduler.step(
                    model_output=model_output,
                    timestep=t,
                    sample=images,
                    generator=step_generator,
                )["prev_sample"]

            if mask is not None:
                if mask_start > 0:
                    images[:, :, :, :mask_start] = mask[:, step, :, :mask_start]
                if mask_end > 0:
                    images[:, :, :, -mask_end:] = mask[:, step, :, -mask_end:]

        if self.vqvae is not None:
            # 0.18215 was scaling factor used in training to ensure unit variance
            images = 1 / 0.18215 * images
            images = self.vqvae.decode(images)["sample"]

        # images = (images / 2 + 0.5).clamp(0, 1) # 2,22,32,64
        images = (images / 2 + 0.5)
        images = images.cpu().numpy()
        # channels = images.shape[1]
        # images = (images * 255).round().astype("uint8")
        # select bs 1 for visualization 22 channels
        spec_images = []
        topoimages = []
        for index, featuremap in enumerate(images):
            if target_ids is not None:
                if featuremap.shape[0] == 8:
                    for channel_feature_map in featuremap:
                        temp_img = visualize_feature_map_withtoken(channel_feature_map.transpose(1,0), target_ids[index])
                        spec_images.append(temp_img)
                else:
                    temp_img = visualize_feature_map_withtoken(featuremap[0], target_ids[index])
                    spec_images.append(temp_img)
            else:
                temp_img = visualize_feature_map(featuremap[0])
                spec_images.append(temp_img)
        to_remove = [8, 14, 17, 21, 25, 43, 48, 49, 56, 63, 68, 73, 81, 88, 94, 99, 107, 113, 119, 120, 126, 127, 128]
        # get from the zuco authors
        my_channels_indiceslist = [x for x in range(0, 128) if x+1 not in to_remove]
        for index, featuremap in enumerate(images):
            if featuremap.shape[0] == 8:
                featuremap = normalize_feature_map(featuremap)
                topolist = create_topo_heat_maps(featuremap, target_ids[index] , my_channels_indiceslist, skip_special_tokens=True)
                topoimages.append(topolist)
            else:
                pass

        # for channel_index in range():
        

        if not return_dict:
            return spec_images, (500, topoimages)

        return BaseOutput(**wavePipelineOutput(np.array(waves)[:, np.newaxis, :]), **ImagePipelineOutput(images))

    @torch.no_grad()
    def encode(self, images: List[Image.Image], steps: int = 50) -> np.ndarray:
        """Reverse step process: recover noisy image from generated image.

        Args:
            images (`List[PIL Image]`): list of images to encode
            steps (`int`): number of encoding steps to perform (defaults to 50)

        Returns:
            `np.ndarray`: noise tensor of shape (batch_size, 1, height, width)
        """

        # Only works with DDIM as this method is deterministic
        assert isinstance(self.scheduler, DDIMScheduler)
        self.scheduler.set_timesteps(steps)
        sample = np.array(
            [np.frombuffer(image.tobytes(), dtype="uint8").reshape((1, image.height, image.width)) for image in images]
        )
        sample = (sample / 255) * 2 - 1
        sample = torch.Tensor(sample).to(self.device)

        for t in self.progress_bar(torch.flip(self.scheduler.timesteps, (0,))):
            prev_timestep = t - self.scheduler.num_train_timesteps // self.scheduler.num_inference_steps
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else self.scheduler.final_alpha_cumprod
            )
            beta_prod_t = 1 - alpha_prod_t
            model_output = self.unet(sample, t)["sample"]
            pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * model_output
            sample = (sample - pred_sample_direction) * alpha_prod_t_prev ** (-0.5)
            sample = sample * alpha_prod_t ** (0.5) + beta_prod_t ** (0.5) * model_output

        return sample

    @staticmethod
    def slerp(x0: torch.Tensor, x1: torch.Tensor, alpha: float) -> torch.Tensor:
        """Spherical Linear intERPolation

        Args:
            x0 (`torch.Tensor`): first tensor to interpolate between
            x1 (`torch.Tensor`): seconds tensor to interpolate between
            alpha (`float`): interpolation between 0 and 1

        Returns:
            `torch.Tensor`: interpolated tensor
        """

        theta = acos(torch.dot(torch.flatten(x0), torch.flatten(x1)) / torch.norm(x0) / torch.norm(x1))
        return sin((1 - alpha) * theta) * x0 / sin(theta) + sin(alpha * theta) * x1 / sin(theta)
