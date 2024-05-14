import torch
from torch import nn
import torch.nn.functional as F


class CReLU(nn.Module):
    """Concatenated ReLU activation function, as defined in
    the continual loss of plasticity paper.
    """

    def __init__(self):
        super(CReLU, self).__init__()

    def forward(self, x):
        concatenated = torch.cat((F.relu(x), F.relu(-x)), 1)
        return concatenated


class Abs(nn.Module):
    """Absolute value activation function"""

    def __init__(self):
        super(Abs, self).__init__()

    def forward(self, x):
        return torch.abs(x)


class FTA(nn.Module):
    """
    ### Fuzzy Tiling Activations (FTA)
    """

    def __init__(self, lower_limit=-20, upper_limit=20, delta=2, eta=2):
        """
        :param lower_limit: is the lower limit $l$
        :param upper_limit: is the upper limit $u$
        :param delta: is the bin size $\delta$
        :param eta: is the parameter $\eta$ that detemines the softness of the boundaries.
        """
        super().__init__()
        # Initialize tiling vector
        # $$\mathbf{c} = (l, l + \delta, l + 2 \delta, \dots, u - 2 \delta, u - \delta)$$
        self.c = nn.Parameter(
            torch.arange(lower_limit, upper_limit, delta), requires_grad=False
        )
        # The input vector expands by a factor equal to the number of bins $\frac{u - l}{\delta}$
        self.expansion_factor = len(self.c)
        # $\delta$
        self.delta = delta
        # $\eta$
        self.eta = eta

    def fuzzy_i_plus(self, x: torch.Tensor):
        """
        #### Fuzzy indicator function
        $$I_{\eta,+}(x) = I_+(\eta - x) x + I_+ (x - \eta)$$
        """
        return (x <= self.eta) * x + (x > self.eta)

    def forward(self, z: torch.Tensor):
        # Add another dimension of size $1$.
        # We will expand this into bins.
        z = z.view(*z.shape, 1)

        # $$\phi_\eta(z) = 1 - I_{\eta,+} \big( \max(\mathbf{c} - z, 0) + \max(z - \delta - \mathbf{c}, 0) \big)$$
        z = 1.0 - self.fuzzy_i_plus(
            torch.clip(self.c - z, min=0.0)
            + torch.clip(z - self.delta - self.c, min=0.0)
        )

        # Reshape back to original number of dimensions.
        # The last dimension size gets expanded by the number of bins, $\frac{u - l}{\delta}$.
        return z.view(*z.shape[:-2], -1)
