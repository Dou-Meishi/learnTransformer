import torch


class MyBatchNorm1d(torch.nn.Module):
    """Implement BatchNorm1d."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = torch.nn.Parameter(torch.ones(num_features))
        self.bias = torch.nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        assert x.ndim == 2
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            self.running_mean = (
                self.momentum * mean + (1 - self.momentum) * self.running_mean
            )
            # The unbiased variance is used in the inference mode
            # see the original paper https://arxiv.org/pdf/1502.03167
            self.running_var = (
                self.momentum * var * x.size(0) / (x.size(0) - 1)
                + (1 - self.momentum) * self.running_var
            )
        else:
            mean = self.running_mean
            var = self.running_var

        x = (x - mean) / (var + self.eps).sqrt()
        x = x * self.weight + self.bias
        return x

    def load_from_pytorch_module(self, model: torch.nn.BatchNorm1d):
        """Load weights from a PyTorch module."""
        self.load_state_dict(
            {
                "weight": model.weight,
                "bias": model.bias,
                "running_mean": model.running_mean,
                "running_var": model.running_var,
            }
        )
