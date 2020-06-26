from .__init__ import *
from ..utils.utils import parse_numbers


def get_config(estimator):
    if estimator == 'pathwise':
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

    elif any([e in estimator for e in ['reinforce', 'vimco', 'ovis']]):
        reinforce_args = {'tau': 0, 'zgrads': False}

        if 'reinforce' in estimator:
            Estimator = Reinforce
            config = reinforce_args

        elif 'vimco' in estimator:

            if '-geometric' in estimator:
                arithmetic = False
            elif 'arithmetic' in estimator:
                arithmetic = True
            else:
                raise ValueError(f"Estimator arg = {estimator} must contain `-arithmetic` or `-geometric`")

            Estimator = Vimco
            config = {**reinforce_args, 'arithmetic': arithmetic}

        elif 'ovis' in estimator:

            if "-S" in estimator:  # parse `-S` : number of auxiliary particles
                auxiliary_samples = int(eval([s for s in estimator.split("-") if 'S' in s][0].replace("S", "")))
                Estimator = OvisMonteCarlo
                config = {**reinforce_args, 'auxiliary_samples': auxiliary_samples}
            elif "-gamma" in estimator:  # parse `-gamma` : parameter of the unified asymptotic approximation
                gamma = float(eval([s for s in estimator.split("-") if 'gamma' in s][0].replace("gamma", "")))
                Estimator = OvisAsymptotic
                config = {**reinforce_args, 'gamma': gamma}

        else:
            raise ValueError(
                f"Unknown reinforce-like estimator {estimator}, valid base identifiers are [reinforce, vimco, copt]")

    elif estimator == 'gs':
        Estimator = VariationalInference
        config = {'tau': 0.5, 'zgrads': True}

    elif estimator == 'st-gs':
        Estimator = VariationalInference
        config = {'tau': 0, 'zgrads': True}

    elif 'tvo' in estimator:
        # argument for automatic partition
        if "-config1" in estimator:
            partition_name = "config1"
        elif "-config2" in estimator:
            partition_name = "config2"
        else:
            partition_name = None

        # number of partitions `-part*`
        partition_args = [arg for arg in estimator.split('-') if 'part' in arg]
        num_partition = parse_numbers(partition_args[0])[0] if len(partition_args) else 2

        # parse `log_beta_min` from as - log beta_min
        partition_args = [arg for arg in estimator.split('-') if 'beta' in arg]
        log_beta_min = - parse_numbers(partition_args[0])[0] if len(partition_args) else -10

        # integration type [`-left`, `-right`, `trapz`]
        _integrations = ['left', 'right', 'trapz']
        integration = [x for x in _integrations if x in estimator.split("-")]
        integration = integration[0] if len(integration) else "left"

        # define config and Estimator
        Estimator = ThermoVariationalObjective
        config = {'tau': 0, 'zgrads': False, 'num_partition': num_partition, 'integration': integration,
                  'partition_name': partition_name, 'log_beta_min': log_beta_min}

    else:
        raise ValueError(f"Unknown estimator {estimator}.")

    return Estimator, config
