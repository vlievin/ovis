# Rewiewing the existing estimators for discrete latent variable VAEs

## Estimators

* REINFORCE
* REINFORCE with baseline
* REINFORCE with baseline and Importance-Weighted objective)
* VIMCO
* Concrete/Gumbel-Softmax
* Straight-Through Concrete/Gumbel-Softmax
* Optimal Covariance Baseline (see overleaf doc)
* RELAX ([TODO]: learn temperature, [TODO]: use iw samples) 
* ! [TODO] TVO
* [TODO] VQ 
* [TODO] REBAR (special case of RELAX)

## Datasets

* Shapes Dataset
* Binarized MNIST
* Omniglot
* Fashion MNIST
* [TODO] BookCorpus Dataset

## Experiments

* Binary Images Modelling
    * nsamples x CovBaseline vs. Vimco (Categorical)
    * nsamples x CovBaseline vs. Vimco vs Pathwise (Gaussian)
    * [TODO] nsamples x CovBaseline vs. TVO
    * [TODO] optimal budget: iw vs. mc samples

* [TODO] Bernoulli toy
    * RELAX: L(θ) = Ep(b|θ) [(b − 0.499)**2]
    * visualize gradients, parameter space, ... 
    
* [TODO] Attend, Infer and Repeat
    * Covbaseline vs. Vimco vs. Reinforce+baseline vs. TVO

* [TODO] Langugage as a Latent Variable
    * Covbaseline vs. Vimco vs. Reinforce+baseline vs. TVO
    
   
* Secondary
    * kdim: effect of the query model 
    * [TODO] nz, kz: best configuration