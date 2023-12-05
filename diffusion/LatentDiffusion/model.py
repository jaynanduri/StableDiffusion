import config
from torch import autocast
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer


class StableDiffusion:
    def __init__(self):
        self.vae = AutoencoderKL.from_pretrained(config.VAE_MODEL_PATH, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(config.UNET_PATH, subfolder="unet")
        self.tokenizer = CLIPTokenizer.from_pretrained(config.CLIP_TOKENIZER_PATH)
        self.text_encoder = CLIPTextModel.from_pretrained(config.CLIP_ENCODER_PATH)


if __name__ == "__main__":
    diffuser = StableDiffusion()
    print("test")
