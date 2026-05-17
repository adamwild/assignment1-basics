# python -m cs336_basics.layers.embedding

import torch

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """Constructs an embedding module.

        Args:
            num_embeddings (int): Size of the vocabulary
            embedding_dim (int): Dimension of the embedding vectors, i.e., d_model
            device (torch.device, optional): Device to store the parameters on. Defaults to None.
            dtype (torch.dtype, optional): Data type of the parameters. Defaults to None.
        """
        super().__init__()

        # mu=0, var=1 ; trunc [-3, 3]
        self.embedding_matrix = torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(tensor=self.embedding_matrix, mean=0.0, std=1, a = -3, b= 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Lookup the embedding vectors for the given token IDs.

        Args:
            token_ids (torch.Tensor): Tensor of tokens (integers)

        Returns:
            torch.Tensor: Tensor of the corresponding embedded vectors
        """
        return self.embedding_matrix[token_ids]