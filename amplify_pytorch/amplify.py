from __future__ import annotations

import torch
from torch.nn import Module

from x_transformers import (
    Encoder,
    Decoder,
    Attention,
    TransformerWrapper
)

from vector_quantize_pytorch import FSQ

from vit_pytorch.vit import ViT
from vit_pytorch.extractor import Extractor

from einops import rearrange, pack, unpack

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
        llm: TransformerWrapper | Module,
        vit: ViT,
        decoder: Decoder,
        action_cross_attn_pool_kwargs: dict = dict()
    ):
        super().__init__()

        self.tokenizer = tokenizer

        self.llm = llm
        self.vit = vit

        self.embed = nn.Embedding(tokenizer.codebook_size, decoder.dim)

        self.decoder = decoder

        self.pool_to_actions = Attention(**action_cross_attn_pool_kwargs)

    def forward(
        self,
        motion_data,
        command, # Int['b nc']
        videos,  # Float['b c t h w']
        additional_prepended_embeds
    ):
        batch = motion_data.shape[0]

        token_ids = self.tokenizer.tokenize(motion_data)

        # language

        command_embed = self.llm(command, return_embeds = True)

        # video to image tokens to be prepended

        images = rearange(videos, 'b c t h w -> b t c h w')
        images = rearrange(images, 'b t c h w -> (b t) c h w')

        image_tokens = self.vit(images)

        prepended_embeds, _ = pack((
            command_embed,
            image_tokens
            additional_prepended_embeds,
        ), 'b * d')

        prepend_len = prepended_embeds.shape[1]

        tokens = self.embed(token_ids)

        inp, target = tokens[:, :-1], tokens[:, 1:]

        logits = self.decoder(
            inp,
            prepend_embeds = prepended_embeds
        )

        logits = logits[:, prepend_len:]

        autoregressive_loss = F.cross_entropy(
            rearrange(logits, 'b n l -> b l n'),
            target,
            ignore_index = -1
        )

        return autoregressive_loss
