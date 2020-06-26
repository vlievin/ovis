import numpy as np


class Schedule():
    def __init__(self, period, init_value, end_value, offset=0, mode='linear'):
        self.offset = offset
        self.period = period
        self.init_value = init_value
        self.end_value = end_value
        self.mode = mode

    def __call__(self, step):
        x = max(0, step - self.offset)
        x = float(x) / self.period

        if self.mode == 'linear':
            x = max(0, min(1, x))
            return self.init_value + x * (self.end_value - self.init_value)

        elif self.mode == 'log':
            x = max(0, min(1, x))
            a = np.log(1 - self.init_value)
            b = np.log(1 - self.end_value)
            x = (1 - x) * a + x * b
            return 1 - np.exp(x)

        elif self.mode == 'sigmoid':
            scale = 3
            t = 2 * scale * (x - 1)
            t = 1 / (1 + np.exp(-t))
            # correction
            t -= 1 / (1 + np.exp(2 * scale)) * (1 - max(0, min(1, x)))
            return self.init_value + t * (self.end_value - self.init_value)

        else:
            raise ValueError(f"Unknown schedule mode = `{self.mode}`")
