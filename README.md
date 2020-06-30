![Optimal Variance Control of the Score Function Gradient Estimator for Importance Weighted Bounds (a.k.a **OVIS**)](.images/ovis-banner.png)

Code for the *Optimal Variance Control of the Score Function Gradient Estimator for Importance Weighted Bounds* (a.k.a **OVIS** of Optimal Variance -- Importance Sampling).

## install as a Package

```bash
pip install git+https://github.com/vlievin/ovis.git
```

## Experiments

### Asymptotic Variance

Anayse the gradients for a simple Gaussian model. Reproduce the figure 1:

```bash
# run the experiment
python manager.py --exp asymptotic-analysis
# produce the figures
python report_asymptotic_variance --exp asymptotic-analysis
# access the results
open reports/asymptotic-variance/
```

Train the Gaussian model:

```bash
# run the experiment
python manager.py --exp gaussian-toy
# produce the figures
python report.py --exp gaussian-toy \
    --keys=dataset,estimator,iw \
    --metrics=train:grads/snr,train:grads/dsnr,valid:gaussian_toy/mse_A,valid:gaussian_toy/mse_b,valid:gaussian_toy/mse_mu \ 
    --detailed_metrics=test:gaussian_toy/mse_A,train:grads/variance,train:grads/snr,train:loss/ess \
    --pivot_metrics=min:test:gaussian_toy/mse_A,min:test:gaussian_toy/mse_b,min:test:gaussian_toy/mse_mu,avg:train:grads/snr
# access the results
open reports/gaussian-toy/
```

### Gaussian Mixture Model

Train a simple Gaussian Mixture model. Reproduce the figure 2:

```bash
# run the experiment
python manager.py --exp gaussian-mixture-model
# produce the figures
python report.py --exp gaussian-mixture-model \
    --keys=dataset,estimator,iw \
    --metrics=test:gmm/posterior_mse,test:gmm/prior_mse,train:grads/variance,train:grads/snr \
    --detailed_metrics=test:gmm/posterior_mse,test:gmm/prior_mse,train:loss/ess,train:grads/variance,train:grads/snr \
    --pivot_metrics=min:test:gmm/posterior_mse,min:test:gmm/prior_mse,mean:train:grads/snr 
# access the results
open reports/gaussian-mixture-model/
```

### Sigmoid Belief Network

Train a 3-layers Sigmoid Belief Network. Reproduce the figure 3 (left):

```bash
# run the experiment
python manager.py --exp sigmoid-belief-network
# produce the figures
[todo: write script]
# access the results
open reports/sigmoid-belief-network/
```

### Gaussian VAE

Train a 1-layer Gaussian VAE. Reproduce the figure 3 (right):

```bash
# run the experiment
python manager.py --exp gaussian-vae
# produce the figures
[todo: write script]
# access the results
open reports/gaussian-vae/
```