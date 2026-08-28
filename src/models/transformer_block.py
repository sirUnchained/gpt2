import torch
from torch import nn

from configs.model_configs import GPT_configs
from .multi_head_attention import MultiHeadAttention
from .normalizers import LayerNormalizer
from .feed_forward import FeedForward


class TransformerBlock(nn.Module):
    """
    ## Transformer Block

    This is the core concept of LLMs. For the first part of this block we are going to use Multi Head Attention like this:

    1. We first get backup from input data for skip connections.
    2. Now the data will be passed from a layer normalizer.
    3. Data will passed into Multi Head Attention module.
    4. We apply some dropout on data.
    5. And finally we just caluclate sum of the bakuped data and current data.

    In second part of this block, we use FeedForward module:
    1. We first get backup from input data for skip connections.
    2. Now the data will be passed from a layer normalizer.
    3. Data will passed into Feed Forward module.
    4. We apply some dropout on data.
    5. And finally we just caluclate sum of the bakuped data and current data.

    ---

    Args:
        cfg (GPT_configs): Configuration object containing model hyperparameters:
            - vocab_size (int): Size of the vocabulary.
            - emb_dim (int): Embedding dimension.
            - context_length (int): Maximum sequence length (context window).
            - drop_rate (float): Dropout probability.
            - n_layers (int): Number of transformer blocks.
            - qkv_bias (bool): Whether to use bias in attention linear layers.
    """

    def __init__(self, cfg: GPT_configs) -> None:
        super().__init__()

        self.MHA = MultiHeadAttention(
            d_in=cfg.emb_dim,
            d_out=cfg.emb_dim,
            context_length=cfg.context_length,
            dropout=cfg.drop_rate,
            n_heads=cfg.n_heads,
            qkv_bias=cfg.qkv_bias,
        )

        self.dropout = nn.Dropout(p=dropout)

        self.norm_1 = LayerNormalizer(emb_dim=cfg.emb_dim)

        self.ff = FeedForward(cfg=cfg)

        self.norm_2 = LayerNormalizer(emb_dim=cfg.emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm_1(x)
        x = self.MHA(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.norm_2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + shortcut

        return x


if __name__ == "__main__":
    cfg = GPT_configs()

    tb = TransformerBlock(cfg=cfg)
    data = torch.rand(3, 4, cfg.emb_dim)

    print(tb(data))
