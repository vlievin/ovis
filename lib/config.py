from .estimators import *
from .structured_estimators import *


def get_config(estimator):
    if estimator == 'vi':
        Estimator = VariationalInference
        config = {'tau': 0, 'zgrads': False}

    elif estimator == 'reinforce':
        Estimator = Reinforce
        config = {'tau': 0, 'zgrads': False}

    elif estimator == 'vimco':
        Estimator = Vimco
        config = {'tau': 0, 'zgrads': False}

    elif estimator == 'gs':
        Estimator = VariationalInference
        config = {'tau': 0.5, 'zgrads': True}

    elif estimator == 'st-gs':
        Estimator = VariationalInference
        config = {'tau': 0, 'zgrads': True}

    elif estimator == 'relax':
        Estimator = Relax
        config = {'tau': 0.5, 'zgrads': True}

    elif estimator == 'covbaseline':
        Estimator = OptCovReinforce
        config = {'tau': 0, 'zgrads': False}

    elif estimator == 'struct-reinforce':
        Estimator = StructuredReinforce
        config = {'tau': 0, 'zgrads': False}

    else:
        raise ValueError(f"Unknown estimator {estimator}.")

    return Estimator, config
