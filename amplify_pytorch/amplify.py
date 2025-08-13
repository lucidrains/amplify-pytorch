from __future__ import annotations

import torch
from torch import nn, cat
import torch.nn.functional as F
from torch.nn import Module, Identity

from x_transformers import (
    Encoder,
    Decoder,
    AttentionPool,
    TransformerWrapper
)

from vector_quantize_pytorch import FSQ

from vit_pytorch.vit import ViT
from vit_pytorch.extractor import Extractor
from vit_pytorch.accept_video_wrapper import AcceptVideoWrapper

from einops import rearrange, repeat, pack, unpack

# helpers

def exists(v):
    return v is not None

# motion tokenizer

class MotionTokenizer(Module):
    def __init__(
        self,
        dim,
        channel_splits = 1,
        codebook_size = 64,
        fsq_kwargs: dict = dict(
            levels = [8, 5, 5, 5]
        )
    ):
        super().__init__()
        self.encoder = Identity()

        self.fsq = FSQ(
            dim = dim,
            num_codebooks = channel_splits,
            **fsq_kwargs
        )

        self.decoder = Identity()

    @property
    def codebook_size(self):
        return self.fsq.codebook_size

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

        recon = self.decoder(quantized)

        recon_loss = F.mse_loss(data, recon)
        return recon_loss

# amplify

# forward and inverse dynamics

class Amplify(Module):
    def __init__(
        self,
        tokenizer: MotionTokenizer,
        llm: TransformerWrapper | Module,
        vit: dict | ViT,
        dim_proprio,
        dim_image_embed,
        action_chunk_size,
        decoder: dict | Decoder,
        dim_action = 20,
        video_time_seq_len = 16,
        inverse_dynamics_transformer_depth = 2,
        action_cross_attn_pool_kwargs: dict = dict(),
        pred_action_loss_weight = 1.
    ):
        super().__init__()

        self.tokenizer = tokenizer

        self.llm = llm

        if isinstance(decoder, dict):
            decoder = Decoder(**decoder)

        dim_model = decoder.dim
        self.motion_sos = nn.Parameter(torch.randn(dim_model))

        self.embed = nn.Embedding(tokenizer.codebook_size, dim_model)

        self.to_proprio = nn.Linear(dim_proprio, dim_model)

        if isinstance(vit, dict):
            vit = ViT(**vit)

        self.vit = Extractor(vit, return_embeddings_only = True)

        self.accept_video_vit = AcceptVideoWrapper(
            self.vit,
            add_time_pos_emb = True,
            output_pos_add_pos_emb = 0,
            time_seq_len = video_time_seq_len,
            dim_emb = dim_image_embed,
            proj_embed_to_dim = dim_model
        )

        self.decoder = decoder

        self.to_logits = nn.Linear(dim_model, tokenizer.codebook_size, bias = False)

        self.pool_to_actions = AttentionPool(
            dim = dim_model,
            num_pooled_tokens = action_chunk_size,
            dim_context = dim_model,
            use_transformer_blocks = True,
            depth = inverse_dynamics_transformer_depth,
            **action_cross_attn_pool_kwargs
        )

        self.action_shape = (action_chunk_size, dim_action)

        self.pred_action_loss_weight = pred_action_loss_weight
        self.to_action_pred = nn.Linear(dim_model, dim_action, bias = False)

    def forward(
        self,
        motion_data,
        commands, # Int['b nc']
        videos,  # Float['b c t h w']
        proprio, # Float['b dp']
        additional_prepended_embeds = None,
        actions = None,
        return_loss_breakdown = False
    ):
        batch = motion_data.shape[0]

        token_ids = self.tokenizer.tokenize(motion_data)

        # language

        command_embed = self.llm(commands, return_embeddings = True)

        # forward dynamics

        # video to image tokens to be prepended

        image_tokens = self.accept_video_vit(videos)
        image_tokens = rearrange(image_tokens, 'b t n d -> b (t n) d')

        if not exists(additional_prepended_embeds):
            additional_prepended_embeds = command_embed[:, 0:0]

        prepended_embeds, _ = pack((
            command_embed,
            image_tokens,
            additional_prepended_embeds,
        ), 'b * d')

        prepend_len = prepended_embeds.shape[1]

        motion_tokens = self.embed(token_ids)

        # add motion start token

        motion_sos = repeat(self.motion_sos, 'd -> b 1 d', b = motion_tokens.shape[0])

        motion_tokens = cat((motion_sos, motion_tokens), dim = 1)

        # pack additional embeds, say touch

        decoder_input, packed_shape = pack((prepended_embeds, motion_tokens), 'b * d')

        # autoregressive transformer

        embeds = self.decoder(decoder_input)

        _, motion_tokens_attended = unpack(embeds, packed_shape, 'b * d')

        motion_pred_logits = self.to_logits(motion_tokens_attended)

        autoregressive_loss = F.cross_entropy(
            rearrange(motion_pred_logits[:, :-1], 'b n l -> b l n'),
            token_ids.long(),
            ignore_index = -1
        )

        # inverse dynamics, cross attention based pooling

        proprio_tokens = self.to_proprio(proprio)

        embeds, _ = pack((proprio_tokens, embeds), 'b * d')

        pooled = self.pool_to_actions(embeds)

        action_pred = self.to_action_pred(pooled)

        if not exists(actions):
            return action_pred

        assert actions.shape[1:] == self.action_shape, f'expected shape {self.action_shape} but received {tuple(actions.shape)}'

        action_loss = F.l1_loss(action_pred, actions)

        # handle losses

        loss_breakdown = (autoregressive_loss, action_loss)

        total_loss = (
            autoregressive_loss +
            action_loss * self.pred_action_loss_weight
        )

        if not return_loss_breakdown:
            return total_loss

        return total_loss, loss_breakdown

