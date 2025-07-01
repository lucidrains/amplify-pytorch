import torch
from torch.nn import Module

from x_transformers import (
    Encoder,
    Decoder,
    Attention,
    TransformerWrapper
)

from vector_quantize_pytorch import FSQ

# helpers

def exists(v):
    return v is not None

# main class

class MotionTokenizer(Module):
    def __init__(
        self,
        fsq_kwargs: dict = dict()
    ):
        super().__init__()
        self.fsq = FSQ(**fsq_kwargs)

class Amplify(Module):
    def __init__(
        self,
        tokenizer: MotionTokenizer,
        decoder: Decoder,
        action_cross_attn_pool_kwargs: dict = dict()
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.decoder = decoder

        self.pool_to_actions = Attention(**action_cross_attn_pool_kwargs)