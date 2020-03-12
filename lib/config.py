from .optcov_estimator import *
from .structured_estimators import *


def get_config(estimator):
    if estimator == 'pathwise':
        Estimator = VariationalInference
        config = {'tau': 0, 'zgrads': True}

    elif 'reinforce' in estimator:
        Estimator = Reinforce
        mc_estimate = '-mc' in estimator
        config = {'tau': 0, 'zgrads': False, 'mc_estimate': mc_estimate}

    elif 'covbaseline' in estimator:
        Estimator = OptCovReinforce
        mc_estimate = '-mc' in estimator
        nz_estimate = '-nz' in estimator
        if '-arithmetic' in estimator:
            arithmetic = True
        elif '-geometric' in estimator:
            arithmetic = False
        else:
            raise ValueError(f"Estimator arg = {estimator} must contain `-arithmetic` or `-geometric`")
        config = {'tau': 0, 'zgrads': False, 'mc_estimate': mc_estimate, 'nz_estimate': nz_estimate, 'arithmetic': arithmetic}

    elif 'vimco' in estimator:
        Estimator = Vimco
        mc_estimate = '-mc' in estimator
        if '-arithmetic' in estimator:
            arithmetic = True
        elif '-geometric' in estimator:
            arithmetic = False
        else:
            raise ValueError(f"Estimator arg = {estimator} must contain `-arithmetic` or `-geometric`")
        config = {'tau': 0, 'zgrads': False, 'mc_estimate': mc_estimate, 'arithmetic': arithmetic}

    elif estimator == 'gs':
        Estimator = VariationalInference
        config = {'tau': 0.5, 'zgrads': True}

    elif estimator == 'st-gs':
        Estimator = VariationalInference
        config = {'tau': 0, 'zgrads': True}

    elif estimator == 'relax':
        Estimator = Relax
        config = {'tau': 0.5, 'zgrads': True}

    elif estimator == 'struct-reinforce':
        Estimator = StructuredReinforce
        config = {'tau': 0, 'zgrads': False}

    else:
        raise ValueError(f"Unknown estimator {estimator}.")

    return Estimator, config
