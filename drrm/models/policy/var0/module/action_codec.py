import torch
from torch import nn
import torch.nn.functional as F


def swish(x):
    """Swish activation function."""
    return x * torch.sigmoid(x)

class CategorySpecificLinear(nn.Module):
    """Linear layer with category-specific weights and biases for multi-embodiment support."""

    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        """
        Args:
            x: [B, T, input_dim] input tensor
            cat_ids: [B] category/embodiment IDs
        Returns:
            [B, T, hidden_dim] output tensor
        """
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)

    def expand_action_dimension(
        self, old_action_dim, new_action_dim, expand_input=False, expand_output=False
    ):
        """
        Safely expand action dimension with explicit targeting.

        Args:
            old_action_dim: Original action dimension
            new_action_dim: New (larger) action dimension
            expand_input: Whether to expand input dimension (dim=1)
            expand_output: Whether to expand output dimension (dim=2)
        """
        if new_action_dim <= old_action_dim:
            raise ValueError(
                f"New action dim {new_action_dim} must be larger than old action dim {old_action_dim}"
            )

        # Expand input dimension (dim=1) only if explicitly requested AND dimensions match
        if expand_input and self.W.shape[1] == old_action_dim:
            repeat_times = new_action_dim // old_action_dim
            remainder = new_action_dim % old_action_dim

            new_W_parts = [self.W] * repeat_times
            if remainder > 0:
                new_W_parts.append(self.W[:, :remainder, :])

            new_W = torch.cat(new_W_parts, dim=1)
            self.W = nn.Parameter(new_W)

        # Expand output dimension (dim=2) only if explicitly requested AND dimensions match
        if expand_output and self.W.shape[2] == old_action_dim:
            repeat_times = new_action_dim // old_action_dim
            remainder = new_action_dim % old_action_dim

            new_W_parts = [self.W] * repeat_times
            if remainder > 0:
                new_W_parts.append(self.W[:, :, :remainder])

            new_W = torch.cat(new_W_parts, dim=2)
            self.W = nn.Parameter(new_W)

            # Expand bias for output dimension
            if self.b.shape[1] == old_action_dim:
                new_b_parts = [self.b] * repeat_times
                if remainder > 0:
                    new_b_parts.append(self.b[:, :remainder])

                new_b = torch.cat(new_b_parts, dim=1)
                self.b = nn.Parameter(new_b)


class SmallMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        hidden = F.relu(self.layer1(x))
        return self.layer2(hidden)


class CategorySpecificMLP(nn.Module):
    """Two-layer MLP with category-specific weights for multi-embodiment support."""

    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        """
        Args:
            x: [B, T, input_dim] input tensor
            cat_ids: [B] category/embodiment IDs
        Returns:
            [B, T, output_dim] output tensor
        """
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)

    def expand_action_dimension(self, old_action_dim, new_action_dim):
        """
        Expand action dimension by copying weights from existing dimensions.

        Args:
            old_action_dim: Original action dimension
            new_action_dim: New (larger) action dimension
        """
        # self.layer1 does not take action_dim as input, so no expansion needed
        self.layer2.expand_action_dimension(
            old_action_dim, new_action_dim, expand_input=False, expand_output=True
        )


class MultiEmbodimentActionEncoder(nn.Module):
    """Action encoder with multi-embodiment support and sinusoidal positional encoding."""

    def __init__(
        self, 
        action_dim: int,
        hidden_size: int,
        output_dim: int,
        num_embodiments: int
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, output_dim)  # (w -> output_dim)

    def forward(self, actions, cat_ids=None):
        """
        Args:
            actions: [B, T, action_dim] action tensor
            cat_ids: [B,] category/embodiment IDs. Defaults to 0.
        Returns:
            [B, T, hidden_size] encoded action features
        """
        B, T, _ = actions.shape
        if cat_ids is None:
            cat_ids = torch.zeros(B, dtype=torch.long, device=actions.device)

        # 2) Standard action MLP step for shape => (B, T, w)
        x = self.W1(actions, cat_ids)
        x = swish(self.W2(x, cat_ids))
        x = self.W3(x, cat_ids)
        return x

    def expand_action_dimension(self, old_action_dim, new_action_dim):
        """
        Expand action dimension by copying weights from existing dimensions.

        Args:
            old_action_dim: Original action dimension
            new_action_dim: New (larger) action dimension
        """
        # Only W1 takes action_dim as input, so only expand its input dimension
        self.W1.expand_action_dimension(
            old_action_dim, new_action_dim, expand_input=True, expand_output=False
        )


class MultiEmbodimentActionDecoder(nn.Module):
    """Action decoder with multi-embodiment support."""

    def __init__(
        self, 
        input_dim: int,
        hidden_size: int,
        action_dim: int,
        num_embodiments: int
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        self.W1 = CategorySpecificLinear(num_embodiments, input_dim, hidden_size)  # (input_dim -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, action_dim)  # (w -> action_dim)

    def forward(self, features, cat_ids=None):
        """
        Args:
            features: [B, T, input_dim] encoded feature tensor
            cat_ids: [B,] category/embodiment IDs. Defaults to 0.
        Returns:
            [B, T, action_dim] decoded action tensor
        """
        B = features.shape[0]
        if cat_ids is None:
            cat_ids = torch.zeros(B, dtype=torch.long, device=features.device)

        x = self.W1(features, cat_ids)
        x = swish(self.W2(x, cat_ids))
        x = self.W3(x, cat_ids)
        return x

    def expand_action_dimension(self, old_action_dim, new_action_dim):
        """
        Expand action dimension by copying weights from existing dimensions.

        Args:
            old_action_dim: Original action dimension
            new_action_dim: New (larger) action dimension
        """
        # Only W3 outputs to action_dim, so only expand its output dimension
        self.W3.expand_action_dimension(
            old_action_dim, new_action_dim, expand_input=False, expand_output=True
        )