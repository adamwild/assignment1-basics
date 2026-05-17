# python -m cs336_basics.layers.linear

import torch
import math
from einops import einsum

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """Constructs a linear transformation module
        Args:
            in_features (int): final dimension of the input
            out_features (int): final dimension of the output
            device (torch.device, optional): Device to store the parameters on. Defaults to None.
            dtype (torch.dtype, optional): Data type of the parameters. Defaults to None.
        """
        super().__init__()

        # mu=0, var=2 / (d_in + d_out); trunc [-3std, 3std]
        self.W = torch.nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2/(in_features + out_features))
        torch.nn.init.trunc_normal_(tensor=self.W, mean=0.0, std=std, a = -3*std, b= 3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation to the input.

        Args:
            x (torch.Tensor): Vector

        Returns:
            torch.Tensor: Y = W.x
        """
        Y = einsum(self.W, x, "d_out d_in, ... d_in-> ... d_out")
        return Y

if __name__ == '__main__':
    lin = Linear(3, 6, dtype=torch.float32)
    x = torch.linspace(start=0.0, end=1.0, steps=3)
    print(x.size())
    Y = lin.forward(x)
    print(Y)