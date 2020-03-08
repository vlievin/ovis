from .optcov_estimator import *
from .structured_estimators import *


def get_config(estimator):
    if estimator == 'vi':
        Estimator = VariationalInference
        config = {'tau': 0, 'zgrads': False}

    elif 'reinforce' in estimator:
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

    elif 'covbaseline' in estimator:
        Estimator = OptCovReinforce
        mc_estimates = '-mc' in estimator
        nz_estimates = '-nz' in estimator
        config = {'tau': 0, 'zgrads': False, 'mc_estimates': mc_estimates, 'nz_estimates': nz_estimates}

    elif estimator == 'struct-reinforce':
        Estimator = StructuredReinforce
        config = {'tau': 0, 'zgrads': False}

    else:
        raise ValueError(f"Unknown estimator {estimator}.")

    return Estimator, config
