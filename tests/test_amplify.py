import pytest
import torch

def test_tokenizer():
    from amplify_pytorch.amplify import MotionTokenizer

    traj = torch.randn(1, 16, 3, 16, 16, 2)

    tokenizer = MotionTokenizer(dim = 512)

    loss = tokenizer(traj)
    loss.backward()

def test_amplify():

    from amplify_pytorch.amplify import Amplify, MotionTokenizer
    from x_transformers import Decoder, TransformerWrapper

    llm = TransformerWrapper(
        num_tokens = 20000,
        max_seq_len = 1024,
        attn_layers = Decoder(
            dim = 512,
            depth = 6,
            heads = 8
        )
    )

    tokenizer = MotionTokenizer(dim = 32)

    amplify = Amplify(
        tokenizer,
        dim_proprio = 17,
        dim_image_embed = 256,
        action_chunk_size = 20,
        vit = dict(
            image_size = 224,
            patch_size = 14,
            num_classes = 1000,
            heads = 8,
            dim = 256,
            depth = 2,
            mlp_dim = 1024,
        ),
        llm = llm,
        decoder = dict(
            dim = 512,
            depth = 2
        )
    )

    traj = torch.randn(2, 16, 3, 16, 16, 2)

    loss = amplify(
        traj,
        commands = torch.randint(0, 20000, (2, 512)),
        videos = torch.randn(2, 3, 16, 224, 224),
        proprio = torch.randn(2, 17),
        actions = torch.randn(2, 20, 20)
    )

    loss.backward()

    # after much training

    pred_action_chunk = amplify(
        traj,
        commands = torch.randint(0, 20000, (2, 512)),
        videos = torch.randn(2, 3, 16, 224, 224),
        proprio = torch.randn(2, 17),
    )
