from .__init__ import *
from ..utils.utils import parse_numbers


def get_config(estimator_id):
    """
    get the Estimator constructor and the estimator configuration from the estimator identifier.
    :param estimator_id: estimator identifier (e.g. `ovis-gamma1`)
    :return: Estimator, config
    """


    if estimator_id == 'pathwise':
        Estimator = VariationalInference
        config = {'tau': 0, 'zgrads': True}

    elif estimator_id == 'pathwise-vae':
        Estimator = PathwiseVAE
        config = {'tau': 0, 'zgrads': True}

    elif estimator_id == 'pathwise-iwae':
        Estimator = PathwiseIWAE
        config = {'tau': 0, 'zgrads': True}

    elif estimator_id == 'gs':
        Estimator = VariationalInference
        config = {'tau': 0.5, 'zgrads': True}

    elif estimator_id == 'st-gs':
        Estimator = VariationalInference
        config = {'tau': 0, 'zgrads': True}

    elif 'wake-sleep' in estimator_id or 'wake-wake' in estimator_id:
        Estimator = {'wake-sleep': WakeSleep, 'wake-wake': WakeWake}[estimator_id]
        config = {'tau': 0, 'zgrads': False}

    elif any([e in estimator_id for e in ['reinforce', 'vimco', 'ovis']]):
        reinforce_args = {'tau': 0, 'zgrads': False}

        if 'reinforce' in estimator_id:
            Estimator = Reinforce
            config = reinforce_args

        elif 'vimco' in estimator_id:

            if '-geometric' in estimator_id:
                arithmetic = False
            elif 'arithmetic' in estimator_id:
                arithmetic = True
            else:
                raise ValueError(f"Estimator arg = {estimator_id} must contain `-arithmetic` or `-geometric`")

            Estimator = Vimco
            config = {**reinforce_args, 'arithmetic': arithmetic}

        elif 'ovis' in estimator_id:

            if "-S" in estimator_id:  # parse `-S` : number of auxiliary particles
                auxiliary_samples = int(eval([s for s in estimator_id.split("-") if 'S' in s][0].replace("S", "")))
                Estimator = OvisMonteCarlo
                config = {**reinforce_args, 'auxiliary_samples': auxiliary_samples}
            elif "-gamma" in estimator_id:  # parse `-gamma` : parameter of the unified asymptotic approximation
                gamma = float(eval([s for s in estimator_id.split("-") if 'gamma' in s][0].replace("gamma", "")))
                Estimator = OvisAsymptotic
                config = {**reinforce_args, 'gamma': gamma}
            else:
                raise ValueError(
                    f"Ovis estimators should have either of the arguments `-S*` or `-gamma*`. (e.g. ovis-gamma1)")

        else:
            raise ValueError(
                f"Unknown reinforce-like estimator {estimator_id}, valid base identifiers are [reinforce, vimco, copt]")

    elif 'tvo' in estimator_id:
        # argument for automatic partition
        if "-config1" in estimator_id:
            partion_id = "config1"
        elif "-config2" in estimator_id:
            partion_id = "config2"
        else:
            partion_id = None

        # number of partitions `-part*`
        partition_args = [arg for arg in estimator_id.split('-') if 'part' in arg]
        num_partition = parse_numbers(partition_args[0])[0] if len(partition_args) else 2

        # parse `log_beta_min` from as - log beta_min
        partition_args = [arg for arg in estimator_id.split('-') if 'beta' in arg]
        log_beta_min = - parse_numbers(partition_args[0])[0] if len(partition_args) else -10

        # integration type [`-left`, `-right`, `trapz`]
        _integrations = ['left', 'right', 'trapz']
        integration = [x for x in _integrations if x in estimator_id.split("-")]
        integration = integration[0] if len(integration) else "left"

        # define config and Estimator
        Estimator = ThermoVariationalObjective
        config = {'tau': 0, 'zgrads': False, 'num_partition': num_partition, 'integration': integration,
                  'partion_id': partion_id, 'log_beta_min': log_beta_min}

    else:
        raise ValueError(f"Unknown estimator {estimator_id}.")

    return Estimator, config
