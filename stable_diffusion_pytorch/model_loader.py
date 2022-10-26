import torch
from . import Tokenizer, CLIP, Encoder, Decoder, Diffusion
from . import util


def load_clip(device):
    clip = CLIP().to(device)
    clip.load_state_dict(torch.load(util.get_file_path('ckpt/clip.pt')))
    return clip

def load_encoder(device):
    encoder = Encoder().to(device)
    encoder.load_state_dict(torch.load(util.get_file_path('ckpt/encoder.pt')))
    return encoder

def load_decoder(device):
    decoder = Decoder().to(device)
    decoder.load_state_dict(torch.load(util.get_file_path('ckpt/decoder.pt')))
    return decoder

def load_diffusion(device):
    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(torch.load(util.get_file_path('ckpt/diffusion.pt')))
    return diffusion

def preload_models(device):
    return {
        'clip': load_clip(device),
        'encoder': load_encoder(device),
        'decoder': load_decoder(device),
        'diffusion': load_diffusion(device),
    }