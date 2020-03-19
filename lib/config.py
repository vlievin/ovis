from .optcov_estimator import *
from .structured_estimators import *
from .utils import parse_numbers


def get_config(estimator):
    if estimator == 'pathwise':
        Estimator = VariationalInference
        config = {'tau': 0, 'zgrads': True}


    elif any([e in estimator for e in ['reinforce', 'vimco', 'covbaseline']]):

        # parse the pattern `estimator-z_reject{`value`}`
        z_reject_args = [arg for arg in estimator.split('-') if 'z_reject' in arg]

        z_reject = parse_numbers(z_reject_args[0])[0] if len(z_reject_args) else 0

        reinforce_args = {'tau': 0,
                          'zgrads': False,
                          'mc_estimate': '-mc' in estimator,
                          'z_reject': z_reject}

        if 'reinforce' in estimator:
            Estimator = Reinforce
            config = reinforce_args

        elif 'covbaseline' in estimator or 'vimco' in estimator:

            use_outer_samples = '-outer' in estimator
            use_double = not ('-float32' in estimator)
            if '-arithmetic' in estimator:
                arithmetic = True
            elif '-geometric' in estimator:
                arithmetic = False
            else:
                raise ValueError(f"Estimator arg = {estimator} must contain `-arithmetic` or `-geometric`")

            vimco_args = {'arithmetic': arithmetic, 'use_outer_samples': use_outer_samples, 'use_double': use_double}

            if 'vimco' in estimator:
                Estimator = Vimco
                config = {**reinforce_args, **vimco_args}

            elif 'covbaseline' in estimator:
                Estimator = OptCovReinforce
                nz_estimate = '-nz' in estimator
                config = {**reinforce_args, **vimco_args, 'nz_estimate': nz_estimate}

            else:
                raise ValueError(
                    f"Unknown vimco-like estimator {estimator}, valid base identifiers are [vimco, covbaseline]")

        else:
            raise ValueError(
                f"Unknown reinforce-like estimator {estimator}, valid base identifiers are [reinforce, vimco, covbaseline]")



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
