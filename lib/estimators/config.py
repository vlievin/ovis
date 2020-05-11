from .__init__ import *
from lib.utils import parse_numbers

def get_config(estimator):

    if estimator == 'safe-vi':
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


    elif 'wake-sleep' in estimator or 'wake-wake' in estimator:
        Estimator = {'wake-sleep': WakeSleep, 'wake-wake': WakeWake}[estimator]
        config = {'tau': 0, 'zgrads': False}

    elif any([e in estimator for e in ['reinforce', 'vimco', 'copt', 'ww']]) and not 'old' in estimator:
        reinforce_args = {'tau': 0,
                          'zgrads': False}

        if 'reinforce' in estimator:
            Estimator = Reinforce
            config = reinforce_args

        elif 'copt' in estimator or 'vimco' in estimator or 'ww' in estimator:

            if 'copt-uniform' in estimator:
                mode = 'copt-uniform'
            elif 'copt' in estimator:
                mode = 'copt'
            elif 'vimco-geometric' in estimator:
                mode = 'vimco-geometric'
            elif 'vimco' in estimator:
                mode = 'vimco'
            elif 'ww' in estimator:
                mode = 'ww'
            else:
                raise ValueError(f"Unknown estimator mode.")

            # parse `-alphaX`
            if "-alpha" in estimator:
                alpha = eval([s for s in estimator.split("-") if 'alpha' in s][0].replace("alpha", ""))
            else:
                alpha = 1.

            # parse `-truncX`
            if "-trunc" in estimator:
                trunc = eval([s for s in estimator.split("-") if 'trunc' in s][0].replace("trunc", ""))
            else:
                trunc = 0.

            handle_low_ess = '-ess' in estimator
            use_second_largest = '-ess2' in estimator

            Estimator = AirReinforce if "air-" in estimator else VimcoPlus
            config = {'mode': mode, 'alpha': alpha, 'truncation': trunc, 'handle_low_ess': handle_low_ess,
                      'use_second_largest': use_second_largest, **reinforce_args}

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
        # argument for automatic partition
        auto_partition = "-auto" in estimator

        # number of partitions
        partition_args = [arg for arg in estimator.split('-') if 'S' in arg]
        num_partition = parse_numbers(partition_args[0])[0] if len(partition_args) else 2

        # parse `log_beta_min` from as - log beta_min
        partition_args = [arg for arg in estimator.split('-') if 'beta' in arg]
        log_beta_min =  - parse_numbers(partition_args[0])[0] if len(partition_args) else -10


        # integration
        _integrations = ['left', 'right', 'trapz']
        integration = [x for x in _integrations if x in estimator.split("-")]
        integration = integration[0] if len(integration) else "left"

        # define config and Estimator
        Estimator = ThermoVariationalObjective
        config = {'tau': 0, 'zgrads': False, 'num_partition': num_partition, 'integration': integration,
                  'auto_partition': auto_partition, 'log_beta_min': log_beta_min}

    else:
        raise ValueError(f"Unknown estimator {estimator}.")

    return Estimator, config
