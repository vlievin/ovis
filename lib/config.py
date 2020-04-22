from .estimators import *
from .estimators.structured_estimators import StructuredReinforce
from .utils import parse_numbers


def get_config(estimator):
    if estimator == 'factorized-pathwise':
        Estimator = FactorizedVariationalInference
        config = {'tau': 0, 'zgrads': True}

    elif estimator == 'safe-vi':
        Estimator = SafeVariationalInference
        config = {'tau': 0, 'zgrads': True}

    elif estimator == 'pathwise':
        Estimator = VariationalInference
        config = {'tau': 0, 'zgrads': True}

    elif estimator == 'pathwise-vae':
        Estimator = PathwiseVAE
        config = {'tau': 0, 'zgrads': True}


    elif estimator == 'pathwise-iwae':
        Estimator = PathwiseIWAE
        config = {'tau': 0, 'zgrads': True}

    elif any([e in estimator for e in ['reinforce', 'vimco', 'copt']]):
        reinforce_args = {'tau': 0,
                          'zgrads': False}

        if 'reinforce' in estimator:
            Estimator = Reinforce
            config = reinforce_args

        elif 'copt' in estimator or 'vimco' in estimator:
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

            elif 'copt' in estimator:
                Estimator = OptCovReinforce
                uniform_v = '-uniform' in estimator
                zero_v = '-zero' in estimator
                old = '-old' in estimator
                config = {**reinforce_args, **vimco_args, 'uniform_v': uniform_v,
                          'zero_v': zero_v, 'old': old}

            else:
                raise ValueError(
                    f"Unknown vimco-like estimator {estimator}, valid base identifiers are [vimco, copt]")

        else:
            raise ValueError(
                f"Unknown reinforce-like estimator {estimator}, valid base identifiers are [reinforce, vimco, copt]")



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

    elif 'tvo' in estimator:
        partition_args = [arg for arg in estimator.split('-') if 'part' in arg]
        partition = parse_numbers(partition_args[0])[0] if len(partition_args) else 21
        Estimator = ThermoVariationalObjective
        config = {'tau': 0, 'zgrads': False, 'partition': partition}

    else:
        raise ValueError(f"Unknown estimator {estimator}.")

    return Estimator, config
