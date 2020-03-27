# -*- coding: utf-8 -*-
"""
Pytorch implementation of toy example from the paper
Backpropagation Through The Void:
Optimizing Control Variates for Black-box Gradient Estimation

Most of the code is from the Pytorch implementation from the author's
Github: https://github.com/duvenaud/relax/blob/master/pytorch_toy.py

We have added:
- Learning curve for exact gradient
- Calculation of gradient variance (translation from tensorflow to 
    pytorch of implementation 
    https://github.com/duvenaud/relax/blob/master/rebar_toy.py)
- Plotting of learning curves and gradient variance

- TODO: Add e.g. TVO and other missing estimators
- TODO: Replace these estimators with imports from Estimators file?
- TODO: Add parser arguments for easier usage and more flexibility.

"""

from __future__ import absolute_import
from __future__ import print_function

from itertools import product

import argparse
import numpy as np
import torch
from tqdm import tqdm
import pandas

import matplotlib.pyplot as plt

class QFunc(torch.nn.Module):
    '''Control variate for RELAX'''

    def __init__(self, num_latents, hidden_size=10):
        super(QFunc, self).__init__()
        self.h1 = torch.nn.Linear(num_latents, hidden_size)
        self.nonlin = torch.nn.Tanh()
        self.out = torch.nn.Linear(hidden_size, 1)

    def forward(self, z):
        # the multiplication by 2 and subtraction is from toy.py...
        # it doesn't change the bias of the estimator, I guess
        z = self.h1(z * 2. - 1.)
        z = self.nonlin(z)
        z = self.out(z)
        return z


def loss_func(b, t):
    return ((b - t) ** 2).mean(1)


#def _parse_args(args):
#    parser = argparse.ArgumentParser(
#        description='Toy experiment from backpropagation throught the void, '
#        'written in pytorch')
#    parser.add_argument(
#        '--estimator', choices=['reinforce', 'relax', 'rebar'],
#        default='reinforce')
#    parser.add_argument('--rand-seed', type=int, default=42)
#    parser.add_argument('--iters', type=int, default=5000)
#    parser.add_argument('--batch-size', type=int, default=1)
#    parser.add_argument('--target', type=float, default=.499)
#    parser.add_argument('--num-latents', type=int, default=1)
#    parser.add_argument('--lr', type=float, default=.01)
#    return parser.parse_args(args)


def reinforce(f_b, b, logits, **kwargs):
    log_prob = torch.distributions.Bernoulli(logits=logits).log_prob(b)
    d_log_prob = torch.autograd.grad(
        [log_prob], [logits], grad_outputs=torch.ones_like(log_prob))[0]
    d_logits = f_b.unsqueeze(1) * d_log_prob
    return d_logits

def exact_gradient(logits, target, **kwargs):
    d_logits = ((1-target)**2 - target**2)#*logits# torch.sigmoid(logits)
    return d_logits.detach()

def _get_z_tilde(logits, b, v):
    theta = torch.sigmoid(logits)
    v_prime = v * (b - 1.) * (theta - 1.) + b * (v * theta + 1. - theta)
    z_tilde = logits + torch.log(v_prime) - torch.log1p(-v_prime)
    return z_tilde


def rebar(
        f_b, b, logits, z, v, eta, log_temp, target, backward, loss_func=loss_func,
         **kwargs):
    z_tilde = _get_z_tilde(logits, b, v)
    temp = torch.exp(log_temp).unsqueeze(0)
    sig_z = torch.sigmoid(z / temp)
    sig_z_tilde = torch.sigmoid(z_tilde / temp)
    f_z = loss_func(sig_z, target)
    f_z_tilde = loss_func(sig_z_tilde, target)
    log_prob = torch.distributions.Bernoulli(logits=logits).log_prob(b)
    d_log_prob = torch.autograd.grad(
        [log_prob], [logits], grad_outputs=torch.ones_like(log_prob))[0]
    d_f_z = torch.autograd.grad(
        [f_z], [logits], grad_outputs=torch.ones_like(f_z),
        create_graph=True, retain_graph=True)[0]
    d_f_z_tilde = torch.autograd.grad(
        [f_z_tilde], [logits], grad_outputs=torch.ones_like(f_z_tilde),
        create_graph=True, retain_graph=True)[0]
    diff = f_b.unsqueeze(1) - eta * f_z_tilde.unsqueeze(1)
    d_logits = diff * d_log_prob + eta * (d_f_z - d_f_z_tilde)
    if backward:
        var_loss = (d_logits ** 2).mean()
        var_loss.backward()
    return d_logits.detach()


def relax(f_b, b, logits, z, v, log_temp, q_func, backward, **kwargs):
    z_tilde = _get_z_tilde(logits, b, v)
    temp = torch.exp(log_temp).unsqueeze(0)
    sig_z = torch.sigmoid(z / temp)
    sig_z_tilde = torch.sigmoid(z_tilde / temp)
    f_z = q_func(sig_z)[:, 0]
    f_z_tilde = q_func(sig_z_tilde)[:, 0]
    log_prob = torch.distributions.Bernoulli(logits=logits).log_prob(b)
    d_log_prob = torch.autograd.grad(
        [log_prob], [logits], grad_outputs=torch.ones_like(log_prob))[0]
    d_f_z = torch.autograd.grad(
        [f_z], [logits], grad_outputs=torch.ones_like(f_z),
        create_graph=True, retain_graph=True)[0]
    d_f_z_tilde = torch.autograd.grad(
        [f_z_tilde], [logits], grad_outputs=torch.ones_like(f_z_tilde),
        create_graph=True, retain_graph=True)[0]
    diff = f_b.unsqueeze(1) - f_z_tilde.unsqueeze(1)
    d_logits = diff * d_log_prob + d_f_z - d_f_z_tilde
    if backward:
        var_loss = (d_logits.mean(0) ** 2).mean()
        var_loss.backward()
    return d_logits.detach()


def run_toy_example(estimator, rand_seed, iters, batch_size, target_val, num_latents, lr, var_resolution = 10, log_variance=False):#args=None):
    #args = _parse_args(args)
    #print('Target is {}'.format(args.target))

    target = torch.Tensor(1, num_latents)
    target.fill_(target_val)
    logits = torch.zeros(num_latents, requires_grad=True)
    eta = torch.ones(num_latents, requires_grad=True)
    log_temp = torch.from_numpy(
        np.array([.5] * num_latents, dtype=np.float32))
    log_temp.requires_grad_(True)
    q_func = QFunc(num_latents)
    torch.manual_seed(rand_seed)
    if estimator == 'reinforce':
        estimator = reinforce
        tunable = []
    elif estimator == 'exact':
        estimator = exact_gradient
        tunable = []
        log_variance = False
    elif estimator == 'rebar':
        estimator = rebar
        tunable = [eta, log_temp]
    else:
        estimator = relax
        tunable = [log_temp] + list(q_func.parameters())
    logit_optim = torch.optim.Adam([logits], lr=lr)
    if tunable:
        tune_optim = torch.optim.Adam(tunable, lr=lr)
    else:
        tune_optim = None

    losses = []
    log_vars = []
    for i in tqdm(range(iters)):
        logit_optim.zero_grad()
        if tune_optim:
            tune_optim.zero_grad()
        u = torch.rand(batch_size, num_latents)
        v = torch.rand(batch_size, num_latents)
        z = logits + torch.log(u) - torch.log1p(-u)
        b = z.gt(0.).type_as(z)
        f_b = loss_func(b, target)

        d_logits = estimator(
            f_b=f_b, b=b, u=u, v=v, z=z, target=target, logits=logits,
            log_temp=log_temp, eta=eta, q_func=q_func, backward=True,
        )
        logits.backward(d_logits.mean(0))  # mean of batch
        d_logits = d_logits.numpy()
        logit_optim.step()
        if tune_optim:
            tune_optim.step()
        thetas = torch.sigmoid(logits.detach()).numpy()
        loss = thetas * (1 - target_val) ** 2
        loss += (1 - thetas) * target_val ** 2
        loss = loss.mean()
        #mean = d_logits.mean()
        #std = d_logits.std()
        #log_var = 2*np.log(std+1e-10)
        #if i % 25 == 0:
        #    print(
        #        'Iter: {} Loss: {:.05f} Thetas: {} Mean: {:.03f} Std: {:.03f} '
        #        'Temp: {:.03f}'.format(
        #            i, loss, thetas, mean, std, torch.exp(log_temp).item())
        #    )
        losses.append(loss)
        # log_vars.append(log_var)
        if i % var_resolution == 0 and log_variance:
            n_variance_samples = 1000
            # rebars = []
            # reinforces = []
            gradients = []
            for _ in range(n_variance_samples):
                # rb, re = sess.run([rebar, reinforce])
                u = torch.rand(batch_size, num_latents)
                v = torch.rand(batch_size, num_latents)
                z = logits + torch.log(u) - torch.log1p(-u)
                b = z.gt(0.).type_as(z)
                f_b = loss_func(b, target)
                
                d_logits = estimator(
                f_b=f_b, b=b, u=u, v=v, z=z, target=target, logits=logits,
                log_temp=log_temp, eta=eta, q_func=q_func, backward=False,)
                #rebars.append(rb)
                #reinforces.append(re)
                gradients.append(d_logits)
            #rebars = np.array(rebars)
            #reinforces = np.array(reinforces)
            gradients = np.array(gradients)
            #re_var = np.log(reinforces.var(axis=0))
            #rb_var = np.log(rebars.var(axis=0))
            grad_var = np.log(gradients.var(axis=0)) # +10e-10
            #if use_reinforce:
            #  variances.append(np.mean(re_var))
            #else:
            #  variances.append(np.mean(rb_var))
            log_vars.append(np.mean(grad_var))

            #diffs = np.abs(rebars.mean(axis=0) - reinforces.mean(axis=0))
            #sess.run([rebar_var.assign(rb_var), reinforce_var.assign(re_var), est_diffs.assign(diffs)])

            #print("rebar variance = {}".format(rb_var.mean()))
            #print("reinforce variance = {}".format(re_var.mean()))
            #print("rebar     = {}".format(rebars.mean(axis=0)[0]))
            #print("reinforce = {}\n".format(reinforces.mean(axis=0)[0]))
    return losses, log_vars


if __name__ == '__main__':
    seeds = [13, 64, 47, 23] #, 78, 202, 674, 8890, 267, 562]
    iters = 10000
    var_resolution = 10

    losses_reinforce = []
    losses_rebar = []
    losses_relax = []
    losses_exact = []

    logvar_reinforce = []
    logvar_rebar = []
    logvar_relax = []
    logvar_exact = []
    for seed in seeds:
        loss, logvar = run_toy_example(estimator='reinforce', rand_seed=seed, iters=iters, batch_size=1,
                                                  target_val=.499, num_latents=1, lr=0.01,
                                                  var_resolution=var_resolution, log_variance=True)
        losses_reinforce.append(loss)
        logvar_reinforce.append(logvar)

        loss, logvar = run_toy_example(estimator='rebar', rand_seed=seed, iters=iters, batch_size=1,
                                                  target_val=.499, num_latents=1, lr=0.01,
                                                  var_resolution=var_resolution, log_variance=True)
        losses_rebar.append(loss)
        logvar_rebar.append(logvar)

        loss, logvar = run_toy_example(estimator='relax', rand_seed=seed, iters=iters, batch_size=1,
                                                  target_val=.499, num_latents=1, lr=0.01,
                                                  var_resolution=var_resolution, log_variance=True)
        losses_relax.append(loss)
        logvar_relax.append(logvar)

        loss, _ = run_toy_example(estimator='exact', rand_seed=seed, iters=iters, batch_size=1,
                                                  target_val=.499, num_latents=1, lr=0.01,
                                                  var_resolution=var_resolution, log_variance=True)
        losses_exact.append(loss)

    iterations = range(iters)

    """ Learning curves """
    plt.plot(iterations, np.mean(losses_reinforce, axis=0))
    plt.plot(iterations, np.mean(losses_rebar, axis=0))
    plt.plot(iterations, np.mean(losses_relax, axis=0))
    plt.plot(iterations, np.mean(losses_exact, axis=0), color='black', ls='-.', alpha=0.5)
    plt.legend(loc='upper right', labels=('REINFORCE', 'REBAR', 'RELAX', 'Exact gradient'))
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training curves for toy example')
    plt.show()

    """ Log variance of estimators """
    iterations = range(0, iters, var_resolution)
    plt.plot(iterations, np.mean(logvar_reinforce, axis=0))
    plt.plot(iterations, np.mean(logvar_rebar, axis=0))
    plt.plot(iterations, np.mean(logvar_relax, axis=0))
    # plt.plot(iters, np.mean(logvar_exact, axis=0), color='black', ls='-.', alpha=0.5)
    plt.legend(loc='upper right', labels=('REINFORCE', 'REBAR', 'RELAX'))
    plt.xlabel('Iterations')
    plt.ylabel('Log variance')
    plt.title('Log variance of Gradient Estimates')
    plt.show()

    print(losses_reinforce)
    print(logvar_reinforce)
    print(logvar_rebar)
    print(logvar_relax)
    iterations = np.arange(0, iters, var_resolution)
    print(iterations)

    A = pandas.Series(logvar_reinforce[0], iterations)
    B = pandas.Series(logvar_rebar[0], iterations)
    C = pandas.Series(logvar_relax[0], iterations)
    point_alpha=0.1
    line_alpha=0.8
    window = 50
    print(A)
    print("hallo", A.rolling(window).mean())
    plt.plot(iterations, A.rolling(window).mean(), alpha=line_alpha, label="REINFORCE")
    plt.plot(iterations, B.rolling(window).mean(), alpha=line_alpha,  label="REBAR")
    plt.plot(iterations, C.rolling(window).mean(), alpha=line_alpha,  label="RELAX")
    plt.legend(loc='best')# bbox_to_anchor=(1.0, 0.75))
    plt.ylabel("Log Variance of Gradient Estimates")
    plt.xlabel("Steps")
    plt.xlim([500, iters])
    plt.show()
