import torch
from torch.nn import Module

from x_transformers import (
    Encoder,
    Decoder,
    Attention,
    TransformerWrapper
)

from vector_quantize_pytorch import FSQ

from einops import rearrange

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
        self.encoder = nn.Identity()

        self.fsq = FSQ(**fsq_kwargs)

        self.decoder = nn.Identity()

    def encode(
        self,
        motion_data
    ):
        encoded = self.encoder(motion_data)
        return self.fsq(encoded)

    @torch.no_grad()
    def tokenize(self, motion_data):
        _, token_ids = self.encode(motion_data)
        return token_ids

    def forward(
        self,
        motion_data
    ):
        quantized, indices = self.encode(motion_data)

        recon = self.encoder(quantized)

        recon_loss = F.mse_loss(data, recon)
        return recon_loss

class Amplify(Module):
    def __init__(
        self,
        tokenizer: MotionTokenizer,
        decoder: Decoder,
        action_cross_attn_pool_kwargs: dict = dict()
    ):
        super().__init__()

        self.tokenizer = tokenizer

        self.embed = nn.Embedding(tokenizer.codebook_size, decoder.dim)

        self.decoder = decoder

        self.pool_to_actions = Attention(**action_cross_attn_pool_kwargs)

    def forward(
        self,
        motion_data,
        prepended_embeds
    ):
        token_ids = self.tokenizer.tokenize(motion_data)

        tokens = self.embed(token_ids)

        inp, target = tokens[:, :-1], tokens[:, 1:]

        logits = self.decoder(
            inp,
            prepend_embeds = prepended_embeds
        )

        autoregressive_loss = F.cross_entropy(
            rearrange(logits, 'b n l -> b l n'),
            target,
            ignore_index = -1
        )

        return  autoregressive_loss
