from .estimators import *
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


    elif  'wake-sleep' in estimator or 'wake-wake' in estimator:
        Estimator = {'wake-sleep': WakeSleep, 'wake-wake': WakeWake}[estimator]
        config = {'tau': 0, 'zgrads': False}

    elif any([e in estimator for e in ['reinforce', 'vimco', 'copt']]):
        reinforce_args = {'tau': 0,
                          'zgrads': False}

        if 'reinforce' in estimator:
            Estimator = Reinforce
            config = reinforce_args

        elif not 'old' in estimator and ('copt' in estimator or 'vimco' in estimator):

            if 'copt' in estimator:
                v_k_hat = 'copt'
            if 'vimco' in estimator:
                v_k_hat = 'vimco'

            # parse `-alphaX`
            if "-alpha" in estimator:
                alpha = eval([s for s in estimator.split("-") if 'alpha' in s][0].replace("alpha", ""))
            else:
                alpha = 1.

            # parse `-etaX`
            if "-eta" in estimator:
                eta = eval([s for s in estimator.split("-") if 'eta' in s][0].replace("eta", ""))
            else:
                eta = 1.

            Estimator = VimcoPlus
            config = {'v_k_hat': v_k_hat, 'alpha': alpha, 'eta': eta, **reinforce_args}

        # keep legacy test if other tests are needed
        elif 'old-copt' in estimator or 'old-vimco' in estimator:
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

    elif 'tvo' in estimator:
        partition_args = [arg for arg in estimator.split('-') if 'part' in arg]
        partition = parse_numbers(partition_args[0])[0] if len(partition_args) else 21
        _integrations = ['left', 'right', 'trapz']
        integration = [x for x in _integrations if x in estimator.split("-")]
        integration = integration[0] if len(integration) else "left"
        Estimator = ThermoVariationalObjective
        config = {'tau': 0, 'zgrads': False, 'partition': partition, 'integration': integration}

    else:
        raise ValueError(f"Unknown estimator {estimator}.")

    return Estimator, config
