import os

import torch


class Session():
    """a small class to ease checkpointing and restoring"""
    best_elbo = (-1e20, 0, 0)
    global_step = 0
    epoch = 0
    filename = "session.tar"

    def __init__(self, run_id, logdir, model, estimator, optimizers):
        self.run_id = run_id
        self.logdir = logdir
        self.model = model
        self.estimator = estimator
        self.optimizers = optimizers

    def state_dict(self):
        return {
            'best_elbo': self.best_elbo,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'run_id': self.run_id,
            'model': self.model.state_dict(),
            'estimator': self.estimator.state_dict(),
            **{self.opt_id(k): o.state_dict() for k, o in enumerate(self.optimizers)}
        }

    @staticmethod
    def opt_id(k):
        return f"optimizer_{k + 1}"

    @property
    def path(self):
        return os.path.join(self.logdir, self.filename)

    def save(self):
        torch.save(self.state_dict(), self.path)

    def load(self):
        device = next(iter(self.model.parameters())).device
        checkpoint = torch.load(self.path, map_location=device)
        assert self.run_id == checkpoint['run_id']
        self.best_elbo = checkpoint['best_elbo']
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model'])
        self.estimator.load_state_dict(checkpoint['estimator'])
        for k, o in enumerate(self.optimizers):
            o.load_state_dict(checkpoint[self.opt_id(k)])

    def restore_if_available(self):
        if os.path.exists(self.path):
            self.load()
