import torch

eps = 1e-20


def get_gradients_log_total_variance(estimator, model, x, key_filter='', **config):
    # todo: double check this
    m = None
    sum_var = torch.zeros((1,), dtype=torch.double)
    model.train()
    loss, diagnostics, output = estimator(model, x, backward=False, **config)
    for l in loss:
        l.backward(create_graph=True, retain_graph=True)
        for k, v in model.named_parameters():
            if key_filter in k and v.grad is not None:
                v_grads = v.grad.detach().var(0).double()
                log_v_grads = torch.log(eps + v_grads)
                if m is None:
                    m = log_v_grads.max()
                sum_var += (log_v_grads - m).exp().sum().detach().item()

    return (m + torch.log(eps + sum_var)).float().detach().item()
