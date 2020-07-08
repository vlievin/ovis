from torch import nn, Tensor, zeros

# gradient estimator
from ovis.estimators.config import parse_estimator_id
Estimator, config = parse_estimator_id("ovis-gamma1")
estimator = Estimator(iw=16, **config)

# dataset: sample x ~ Bernoulli(0.5)
from torch.distributions import Bernoulli
dset = Bernoulli(logits=zeros((1000, 10))).sample()

# define a simple Bernoulli VAE
from ovis.models import TemplateModel
class SimpleModel(TemplateModel):
    def __init__(self, xdim, zdim):
        super().__init__()
        self.inference_network = nn.Linear(xdim, zdim)
        self.generative_model = nn.Linear(zdim, xdim)
        self.register_buffer('prior', zeros((1, zdim,)))

    def forward(self, x: Tensor, zgrads: bool = False, **kwargs):
        # q(z|x)
        qz = Bernoulli(logits=self.inference_network(x))
        # z ~ q(z | x)
        z = qz.rsample() if zgrads else qz.sample()
        # p(x)
        pz = Bernoulli(logits=self.prior)
        # p(x|z)
        px = Bernoulli(logits=self.generative_model(z))
        # store z, pz, qz as list for hierarchical models
        return {'px': px, 'z': [z], 'qz': [qz], 'pz': [pz]}

    def sample_from_prior(self, bs: int, **kwargs):
        pz = Bernoulli(logits=self.prior.expand(bs, *self.prior.shape[1:]))
        z = pz.sample()
        px = Bernoulli(logits=self.generative_model(z))
        return {'px': px, 'z': [z], 'pz': [pz]}


# initialize the model
model = SimpleModel(10, 10)

# optimizer
from torch.optim import Adam
optimizer = Adam(model.parameters(), lr=2e-3)

# dataloader
from torch.utils.data import DataLoader
loader = DataLoader(dset, batch_size=10)

# prepare logging directory
from torch.utils.tensorboard import SummaryWriter
import os
tensorboard_writer = SummaryWriter('runs/example/')
if not os.path.exists('runs/example/'):
    os.makedirs('runs/example/')

# training
from ovis.analysis.gradients import get_gradients_statistics
from ovis.utils.utils import Header
from booster import Aggregator
global_step = 0
for epoch in range(1, 3):

    # training
    agg = Aggregator()
    for x in loader:
        global_step += 1
        loss, diagnostics, output = estimator(model, x, backward=False, **config)
        loss.mean().backward()
        optimizer.step()
        optimizer.zero_grad()
        agg.update(diagnostics)

    # get summary from aggregator
    summary = agg.data.to('cpu')

    # analyse the gradients of the parameters of the inference network
    grad_stats, _ = get_gradients_statistics(estimator, model, x, mc_samples=10, key_filter='inference_network')
    summary.update(grad_stats)

    # log data
    summary.log(tensorboard_writer, global_step)

    # print summary
    with Header(f"Epoch = {epoch}"):
        print(summary)
