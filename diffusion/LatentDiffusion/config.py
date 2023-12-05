import torch

TOKEN = "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"
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

VAE_MODEL_PATH = "CompVis/stable-diffusion-v1-4"
CLIP_TOKENIZER_PATH = "openai/clip-vit-large-patch14"
CLIP_ENCODER_PATH = "openai/clip-vit-large-patch14"
UNET_PATH = "CompVis/stable-diffusion-v1-4"
