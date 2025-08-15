from __future__ import annotations

import torch
from torch import nn, cat
import torch.nn.functional as F
from torch.nn import Module, Parameter, Sequential, Identity

from x_transformers import (
    Encoder,
    Decoder,
    AttentionPool,
    CrossAttender,
    TransformerWrapper
)

from vector_quantize_pytorch import FSQ

from vit_pytorch.vit import ViT
from vit_pytorch.extractor import Extractor
from vit_pytorch.accept_video_wrapper import AcceptVideoWrapper

import einx
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, pack, unpack

# ein notation

# b - batch
# t - time
# v - num views
# h - height
# w - width
# c - velocity / position

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def pack_and_inverse(t, pattern):
    packed, packed_shape = pack([t], pattern)

    def inverse_fn(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        out, = unpack(out, packed_shape, inv_pattern)
        return out

    return packed, inverse_fn

def pad_at_dim(
    t,
    pad: tuple[int, int],
    dim = -1,
    value = 0.
):
    if pad == (0, 0):
        return t

    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

# motion tokenizer

def trajectory_to_velocities(
    trajectories # (b t ...)
):
    trajectories = pad_at_dim(trajectories, (1, 0), dim = 1)
    velocities = trajectories[:, 1:] - trajectories[:, :-1]
    return velocities

def velocities_to_trajectory(
    velocities # (b t ...)
):
    return velocities.cumsum(dim = 1)

class MotionTokenizer(Module):
    def __init__(
        self,
        dim,
        channels = 2,
        height = 16,
        width = 16,
        patch_size = 4,
        num_views = 3,
        channel_splits = 1,
        codebook_size = 64,
        max_time_seq_len = 16,
        encoder_kwargs: dict = dict(
            depth = 2,
            attn_dim_head = 64,
            heads = 8
        ),
        decoder_kwargs: dict = dict(
            depth = 2,
            attn_dim_head = 64,
            heads = 8
        ),
        fsq_kwargs: dict = dict(
            levels = [8, 5, 5, 5]
        )
    ):
        super().__init__()
        self.shape = (num_views, height, width, channels)

        # positional embeddings

        self.view_pos_emb = Parameter(torch.randn(num_views, dim) * 1e-2)
        self.time_pos_emb = Parameter(torch.randn(max_time_seq_len, dim) * 1e-2)

        self.max_time_seq_len = max_time_seq_len

        # encoder

        assert divisible_by(height, patch_size)
        assert divisible_by(width, patch_size)

        self.num_height_patches = height // patch_size
        self.num_width_patches = width // patch_size

        dim_patch = channels * patch_size ** 2

        self.patchify = Sequential(
            Rearrange('b t v (h p1) (w p2) c -> b t h w v (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(dim_patch, dim)
        )

        self.encoder = Encoder(
            dim = dim,
            **encoder_kwargs
        )

        # fsq

        self.fsq = FSQ(
            dim = dim,
            num_codebooks = channel_splits,
            **fsq_kwargs
        )

        # decoder

        self.decoder = CrossAttender(
            dim = dim,
            **decoder_kwargs
        )

        self.depatchify = Sequential(
            nn.Linear(dim, dim_patch),
            Rearrange('b t h w v (p1 p2 c) -> b t v (h p1) (w p2) c', p1 = patch_size, p2 = patch_size),
        )

    @property
    def codebook_size(self):
        return self.fsq.codebook_size

    def encode(
        self,
        velocities, # (b t v h w c)
        return_pack_inverse_fn = False
    ):
        times = velocities.shape[1]
        assert times <= self.max_time_seq_len

        patch_tokens = self.patchify(velocities)

        # add positional embeddings

        patch_tokens = einx.add('b t ... v d, t d, v d', patch_tokens, self.time_pos_emb[:times], self.view_pos_emb)

        # add view positional embedding

        patch_tokens, inverse_fn = pack_and_inverse(patch_tokens, 'b * d')

        encoded = self.encoder(patch_tokens)

        output = self.fsq(encoded)

        if not return_pack_inverse_fn:
            return output

        return output, inverse_fn

    @torch.no_grad()
    def tokenize(
        self,
        trajectories # (b t v h w c)
    ):
        assert trajectories.shape[-4:] == self.shape

        velocities = trajectory_to_velocities(trajectories)

        _, token_ids = self.encode(velocities)
        return token_ids

    def forward(
        self,
        trajectories, # (b t v h w c)
        return_recon_trajectories = False,
        return_recons = False
    ):
        batch, times = trajectories.shape[:2]
        assert trajectories.shape[-4:] == self.shape

        velocities = trajectory_to_velocities(trajectories)

        (quantized, indices), inverse_fn = self.encode(trajectories, return_pack_inverse_fn = True)

        # constitute learned queries for detr like decoder
        # also incorporating details from private correspondance with author

        decoder_queries = einx.add('t d, v d -> t v d', self.time_pos_emb[:times], self.view_pos_emb)

        decoder_queries = repeat(decoder_queries, 't v d -> b t v h w d', b = batch, h = self.num_height_patches, w = self.num_width_patches)

        decoder_queries, _ = pack_and_inverse(decoder_queries, 'b * d')

        decoded_tokens = self.decoder(decoder_queries, context = quantized)

        decoded_tokens = inverse_fn(decoded_tokens)

        recon_velocities = self.depatchify(decoded_tokens)

        recon_loss = F.mse_loss(velocities, recon_velocities)

        recons = (recon_velocities,)

        if return_recon_trajectories:
            recon_trajectories = velocities_to_trajectory(recon_velocities)

            recons = (*recons, recon_trajectories)

        if not return_recons:
            return recon_loss

        return recon_loss, recons

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
        trajectories,
        commands, # Int['b nc']
        videos,  # Float['b c t h w']
        proprio, # Float['b dp']
        additional_prepended_embeds = None,
        actions = None,
        return_loss_breakdown = False
    ):
        batch = trajectories.shape[0]

        token_ids = self.tokenizer.tokenize(trajectories)

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
