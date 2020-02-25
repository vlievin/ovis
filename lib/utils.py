import torch


def flatten(x):
    return x.view(x.size(0), -1)


def reduce(x):
    return flatten(x).sum(1)


def log_sum_exp(x, dim=-1, sum_op=torch.sum, eps: float = 1e-12, keepdim=False):
    """
    Uses the LogSumExp (LSE) as an approximation for the sum in a log-domain.
    :param x: Tensor to compute LSE over
    :param dim: dimension to perform operation over
    :param sum_op: reductive operation to be applied, e.g. torch.sum or torch.mean
    :return: LSE
    """
    max, _ = torch.max(x, dim=dim, keepdim=True)
    _max = max if keepdim else max.squeeze(dim)
    return torch.log(sum_op(torch.exp(x - max), dim=dim, keepdim=keepdim) + eps) + _max
