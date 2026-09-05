from torch import nn
import torch


class LayerNormalizer(nn.Module):
    """
    ## Layer Normalization

    Layer normalization is used to mitigate the vanishing and exploding gradients problem by normalizing the activations across the feature dimension for each training example independently.

    > **NOTE**: Normalizing the data means shifting the mean to approximately 0 and scaling the variance to approximately 1.

    ---

    Args:
        emb_dim (int):
            The embedding dimension (number of features) of the input tensor.

    Returns:
        torch.Tensor: The normalized tensor with the same shape as the input.
    """

    def __init__(self, emb_dim: int) -> None:
        super().__init__()

        self.eps = 1e-5
        self.gamma = nn.Parameter(torch.ones(emb_dim), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(emb_dim), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        norm = (x - mean) / torch.sqrt(var + self.eps)

        return self.gamma * norm + self.beta
