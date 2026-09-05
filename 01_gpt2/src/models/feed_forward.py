import torch
from torch import nn

from configs.model_configs import GPT_configs
from .activations import Gelu


class FeedForward(nn.Module):
    """
    ## Feed Forward

    Feed Forward is an important but simple module in transformer block, It is simply two linear layers with an activation.

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

        self.layers = nn.Sequential(
            nn.Linear(
                in_features=cfg.emb_dim, out_features=cfg.emb_dim * 4, bias=cfg.qkv_bias
            ),
            Gelu(),
            nn.Linear(
                in_features=cfg.emb_dim * 4, out_features=cfg.emb_dim, bias=cfg.qkv_bias
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


if __name__ == "__main__":

    cfg = GPT_configs()

    data = torch.rand((3, 10, cfg.emb_dim))

    ff = FeedForward(cfg)

    print(ff(data))
