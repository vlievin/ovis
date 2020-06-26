def covariance(x):
    x = x - x.mean(dim=1, keepdim=True)
    cov = 1 / (x.size(1) - 1) * x.mm(x.t())
    return cov


def cosine(u, v, dim=-1):
    """cosine similarity"""
    return (u * v).sum(dim=dim) / (u.norm(dim=dim, p=2) * v.norm(dim=dim, p=2))


def percentile(x, q=0.5):
    if x is not None:
        assert q < 1
        x = x.view(-1)
        k = int(q * x.shape[0]) + 1
        v, idx = x.kthvalue(k)  # indexing from 1..
        return v


def safe_mean(x):
    if x is not None:
        return x.mean()


class RunningMean():
    def __init__(self):
        self.mean = None
        self.n = 0

    def update(self, x, k=1):
        """use k > 1 if x is averaged over `k` points, k > 1.
        Useful when averaging over mini-batch with different dimensions."""
        if self.mean is None:
            self.mean = x
        else:
            self.mean = self.n / (self.n + k) * self.mean + k / (self.n + k) * x

        self.n += k

    def __call__(self):
        return self.mean


class RunningVariance():
    def __init__(self):
        self.n = 0
        self.Ex = None
        self.Ex2 = None
        self.K = None

    def update(self, x):
        self.n += 1
        if self.K is None:
            self.K = x
            self.Ex = x - self.K
            self.Ex2 = (x - self.K) ** 2
        else:
            self.Ex += x - self.K
            self.Ex2 += (x - self.K) ** 2

    def __call__(self):
        return (self.Ex2 - (self.Ex * self.Ex) / self.n) / (self.n - 1)