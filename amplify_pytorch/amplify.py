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

class Amplify(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError
