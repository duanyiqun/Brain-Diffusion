# based on https://github.com/huggingface/diffusers/blob/main/examples/train_unconditional.py

import argparse
import os
import pickle
import random
from pathlib import Path
from typing import Optional
import cv2

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset, load_from_disk
from diffusers import (AutoencoderKL, DDIMScheduler, DDPMScheduler,
                       UNet2DConditionModel, UNet2DModel)
from diffusers.optimization import get_scheduler
from braindiffusion.utils.wave_spectron import Spectro, Spectro_STFT
# from braindiffusion.utils.wave_spectron import Spectro_STFT as Spectro
from diffusers.training_utils import EMAModel
from huggingface_hub import HfFolder, Repository, whoami
from librosa.util import normalize
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from tqdm.auto import tqdm

from braindiffusion.pipeline_freqmap_diffusion import WaveDiffusionPipeline

logger = get_logger(__name__)


def get_full_repo_name(model_id: str,
                       organization: Optional[str] = None,
                       token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def main(args):
    output_dir = os.environ.get("SM_MODEL_DIR", None) or args.output_dir
    logging_dir = os.path.join(output_dir, args.logging_dir)
    visualization_dir = os.path.join(output_dir, args.visualization_dir)
    os.makedirs(visualization_dir, exist_ok=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        # log_with="tensorboard",
        log_with="wandb",
        logging_dir=logging_dir,  
    )

    if args.dataset_name is not None:
        if os.path.exists(args.dataset_name):
            dataset = load_from_disk(
                args.dataset_name,
                storage_options=args.dataset_config_name)["train"]
        else:
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
                use_auth_token=True if args.use_auth_token else None,
                split="train",
            )
    else:
        dataset = load_dataset(
            "imagefolder",
            data_dir=args.train_data_dir,
            cache_dir=args.cache_dir,
            split="train",
        )
    # Determine image resolution
    args.original_shape = tuple(int(x) for x in args.original_shape.split(","))
    args.force_rescale = tuple(int(x) for x in args.force_rescale.split(","))
    if args.stft:
        print(np.frombuffer(dataset[0]["image"], dtype=np.float64).reshape(args.original_shape).shape)
        resolution = np.frombuffer(dataset[0]["image"], dtype=np.float64).reshape(args.original_shape).shape
        # resolution = (resolution[0], int(resolution[1]//1.2), int(resolution[2]//1.5))
        # resolution = dataset[0]["image"].height, dataset[0]["image"].width
        max_freq = args.max_freq
        # assert max_freq < args.original_shape[2]
        resolution = np.frombuffer(dataset[0]["image"], dtype=np.float64).reshape(args.original_shape).shape
        resolution = (resolution[0], int(resolution[1]), max_freq)
        if args.force_rescale is not None:
            if resolution[1] > args.force_rescale[1]:
                resolution = (resolution[0], args.force_rescale[1], resolution[2])

        augmentations = Compose([
            lambda x: np.frombuffer(x, dtype=np.float64).reshape(args.original_shape)[:,:,:max_freq].transpose().transpose(1,0,2).astype(np.float32), # convert bytes to numpy original 
            lambda x: np.abs(x), # resize to original
            lambda x: cv2.resize(x, (resolution[2], resolution[1])), # resize to original
            # lambda x: print(x.shape), # resize to original
            ToTensor(),
            Normalize([0.5], [0.5]).float(),
        ])
    else:
        print(np.frombuffer(dataset[0]["image"], dtype=np.float64).reshape(args.original_shape).shape)
        resolution = np.frombuffer(dataset[0]["image"], dtype=np.float64).reshape(args.original_shape).shape
        # resolution = (resolution[0], int(resolution[1]//1.2), int(resolution[2]//1.5))
        # resolution = dataset[0]["image"].height, dataset[0]["image"].width
        max_freq = args.max_freq
        min_freq = args.min_freq
        # assert max_freq < args.original_shape[2]
        resolution = np.frombuffer(dataset[0]["image"], dtype=np.float64).reshape(args.original_shape).shape
        resolution = (resolution[0], int(resolution[1]), max_freq)

        augmentations = Compose([
            lambda x: np.frombuffer(x, dtype=np.float64).reshape(args.original_shape)[:,:,min_freq:max_freq+min_freq].transpose().transpose(1,0,2).astype(np.float32), # convert bytes to numpy original 
            lambda x: np.abs(x), # resize to original
            # lambda x: print(x.shape), # resize to original
            ToTensor(),
            Normalize([0.5], [0.5]).float(),
        ])

    def transforms(examples):
        if args.vae is not None and vqvae.config["in_channels"] == 3:
            images = [
                augmentations(image.convert("RGB"))
                for image in examples["image"]
            ]
        else:
            images = [augmentations(image) for image in examples["image"]]
        if args.encodings is not None:
            encoding = [encodings[str(target_ids)][:int(resolution[1]),:] for target_ids in examples["target_ids"]]
            
            return {"input": images, "encoding": encoding, "target_ids": examples["target_ids"]}
        return {"input": images}

    dataset.set_transform(transforms)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True)

    if args.encodings is not None:
        encodings = pickle.load(open(args.encodings, "rb"))

    vqvae = None
    if args.vae is not None:
        try:
            vqvae = AutoencoderKL.from_pretrained(args.vae)
        except EnvironmentError:
            vqvae = WaveDiffusionPipeline.from_pretrained(args.vae).vqvae
        # Determine latent resolution
        with torch.no_grad():
            latent_resolution = vqvae.encode(
                torch.zeros((1, 1) +
                            resolution)).latent_dist.sample().shape[2:]

    if args.from_pretrained is not None:
        pipeline = WaveDiffusionPipeline.from_pretrained(args.from_pretrained)
        spectro = pipeline.Spectro
        model = pipeline.unet
        if hasattr(pipeline, "vqvae"):
            vqvae = pipeline.vqvae

    else:
        if args.encodings is None:
            model = UNet2DModel(
                sample_size=resolution if vqvae is None else latent_resolution,
                in_channels=args.original_shape[0]
                if vqvae is None else vqvae.config["latent_channels"],
                out_channels=args.original_shape[0]
                if vqvae is None else vqvae.config["latent_channels"],
                layers_per_block=2,
                block_out_channels=(128, 128, 256, 256, 512, 512),
                down_block_types=(
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "AttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "AttnUpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
            )

        else:
            model = UNet2DConditionModel(
                sample_size=resolution if vqvae is None else latent_resolution,
                in_channels=args.original_shape[0]
                if vqvae is None else vqvae.config["latent_channels"],
                out_channels=args.original_shape[0]
                if vqvae is None else vqvae.config["latent_channels"],
                layers_per_block=2,
                block_out_channels=(128, 256, 512, 512),
                down_block_types=(
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                ),
                cross_attention_dim=list(encodings.values())[0].shape[-1],
            )

    if args.scheduler == "ddpm":
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.num_train_steps)
    else:
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=args.num_train_steps)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs) //
        args.gradient_accumulation_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler)

    ema_model = EMAModel(
        getattr(model, "module", model),
        inv_gamma=args.ema_inv_gamma,
        power=args.ema_power,
        max_value=args.ema_max_decay,
    )

    if args.push_to_hub:
        if args.hub_model_id is None:
            repo_name = get_full_repo_name(Path(output_dir).name,
                                           token=args.hub_token)
        else:
            repo_name = args.hub_model_id
        repo = Repository(output_dir, clone_from=repo_name)

    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(args.wandb_projects)

    if args.stft:
        print("setting spectro to Spectro_STFT")
        spectro = Spectro_STFT(
            x_res=resolution[1],
            y_res=resolution[0],
            hop_length=args.hop_length,
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            eeg_channels=args.eeg_channels,
        )
    else:
        print("setting spectro to spectro")
        spectro = Spectro(
            x_res=resolution[1],
            y_res=resolution[0],
            hop_length=args.hop_length,
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            eeg_channels=args.eeg_channels,
        )

    
    global_step = 0
    for epoch in range(args.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader),
                            disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        if epoch < args.start_epoch:
            for step in range(len(train_dataloader)):
                optimizer.step()
                lr_scheduler.step()
                progress_bar.update(1)
                global_step += 1
            if epoch == args.start_epoch - 1 and args.use_ema:
                ema_model.optimization_step = global_step
            continue
        
        ########`  TRAINING LOOP  `########

        model.train()
        for step, batch in enumerate(train_dataloader):
            if step > 1 and args.debug:
                break
            clean_images = batch["input"]
            if args.debug:
                print("clean_images.shape")
                print(clean_images.shape)

            # print(clean_images.shape)
            if vqvae is not None:
                vqvae.to(clean_images.device)
                with torch.no_grad():
                    clean_images = vqvae.encode(
                        clean_images).latent_dist.sample()
                # Scale latent images to ensure approximately unit variance
                clean_images = clean_images * 0.18215

            # Sample noise that we'll add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bsz = clean_images.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.num_train_timesteps,
                (bsz, ),
                device=clean_images.device,
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise,
                                                     timesteps)
            if args.debug:
                for channel_index, image in enumerate(clean_images.cpu().numpy()[0]):
                    spectro.plot_spectrogram(image, save_fig='./clean_image{}.png'.format(channel_index)),

            with accelerator.accumulate(model):
                # Predict the noise residual
                if args.encodings is not None:
                    noise_pred = model(noisy_images, timesteps,
                                       batch["encoding"])["sample"]
                else:
                    noise_pred = model(noisy_images, timesteps)["sample"]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                if args.use_ema:
                    ema_model.step(model)
                optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            if args.use_ema:
                logs["ema_decay"] = ema_model.decay
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            if ((epoch + 1) % args.save_model_epochs == 0
                    or (epoch + 1) % args.save_images_epochs == 0
                    or epoch == args.num_epochs - 1):
                unet = accelerator.unwrap_model(model)
                if args.use_ema:
                    ema_model.copy_to(unet.parameters())
                pipeline = WaveDiffusionPipeline(
                    vqvae=vqvae,
                    unet=unet,
                    Spectro=spectro,
                    scheduler=noise_scheduler,
                )

            if (
                    epoch + 1
            ) % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                pipeline.save_pretrained(output_dir)

                # save the model
                if args.push_to_hub:
                    repo.push_to_hub(
                        commit_message=f"Epoch {epoch}",
                        blocking=False,
                        auto_lfs_prune=True,
                    )

            if (epoch + 1) % args.save_images_epochs == 0:
                generator = torch.Generator(
                    device=clean_images.device).manual_seed(42)

                if args.encodings is not None:
                    random.seed(42)
                    target_ids = random.sample(list(encodings.keys()),
                                      args.eval_batch_size)
                    if args.debug:
                        print(target_ids)
                    encoding = [encodings[key][:int(resolution[1]),:] for key in target_ids]
                    encoding = torch.stack(encoding).to(
                                          clean_images.device)
                    # print(encoding.shape)
                    target_ids = [eval(key) for key in target_ids]
                        
                else:
                    encoding = None

                # run pipeline in inference (sample random noise and denoise)
                images, (sample_rate, waves) = pipeline(
                    generator=generator,
                    batch_size=args.eval_batch_size,
                    return_dict=False,
                    encoding=encoding,
                    target_ids=target_ids,
                )

                # denormalize the images and save to tensorboard
                if args.save_image_tensorboard:
                    images = np.array([
                        np.frombuffer(image.tobytes(), dtype="uint8").reshape(
                            (len(image.getbands()), image.height, image.width))
                        for image in images
                    ])
                    accelerator.trackers[0].writer.add_images(
                        "generated spetro images of 22 channels", images, epoch)

                for index, image in enumerate(images):
                    # print(type(image))
                    os.makedirs(os.path.join(visualization_dir, f"epoch{epoch}"), exist_ok=True)
                    image.save(os.path.join(os.path.join(visualization_dir, f"epoch{epoch}"), f"featuremap_images:{epoch}_bs{index}.png"))    

        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dataset_name", type=str, default='dataset/zuco/freqmap_32_840')
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data.",
    )
    parser.add_argument("--output_dir", type=str, default="models/zuco-freq_feature")
    parser.add_argument("--overwrite_output_dir", type=bool, default=False)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_images_epochs", type=int, default=1)
    parser.add_argument("--save_model_epochs", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.95)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--use_ema", type=bool, default=True)
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0)
    parser.add_argument("--ema_power", type=float, default=3 / 4)
    parser.add_argument("--ema_max_decay", type=float, default=0.9999)
    parser.add_argument("--push_to_hub", type=bool, default=False)
    parser.add_argument("--use_auth_token", type=bool, default=False)
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_private_repo", type=bool, default=False)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--visualization_dir", type=str, default="visualizations")
    parser.add_argument("--original_shape", type=str, default="1,32,840")
    parser.add_argument("--force_rescale", type=str, default="1,32,840")
    parser.add_argument("--max_freq", type=int, default=64)
    parser.add_argument("--min_freq", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    # parser.add_argument("--debug", type=bool, default=True)
    parser.add_argument("--save_image_tensorboard", type=bool, default=False)
    parser.add_argument("--wandb_projects", type=str, default='braindiffusion')
    parser.add_argument("--eeg_channels", type=int, default=1)
    parser.add_argument("--stft", action="store_true")
    parser.add_argument
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."),
    )
    parser.add_argument("--hop_length", type=int, default=50)
    parser.add_argument("--sample_rate", type=int, default=500)
    parser.add_argument("--n_fft", type=int, default=100)
    parser.add_argument("--from_pretrained", type=str, default=None)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--num_train_steps", type=int, default=1000)
    parser.add_argument("--scheduler",
                        type=str,
                        default="ddim",
                        help="ddpm or ddim")
    parser.add_argument(
        "--vae",
        type=str,
        default=None,
        help="pretrained VAE model for latent diffusion",
    )
    parser.add_argument(
        "--encodings",
        type=str,
        default="./dataset/zuco/spectro_dp/text_encodings_train.pt",
        help="picked dictionary mapping audio_file to encoding",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError(
            "You must specify either a dataset name from the hub or a train data directory."
        )

    main(args)
