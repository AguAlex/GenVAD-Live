from functools import partial
from torch import nn
from model.mae import MaskedAutoencoderCvT

def mae_cvt_patch16(**kwargs):
    # Model mult mai mic (Micro-Transformer)
    model = MaskedAutoencoderCvT(
        patch_size=16, embed_dim=128, depth=2, num_heads=2,
        decoder_embed_dim=64, decoder_depth=1, decoder_num_heads=2,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_cvt_patch8(**kwargs):
    model = MaskedAutoencoderCvT(
        patch_size=8, embed_dim=128, depth=2, num_heads=2,
        decoder_embed_dim=64, decoder_depth=1, decoder_num_heads=2,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model