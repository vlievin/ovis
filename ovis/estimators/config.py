from typing import *

from .__init__ import *
from ..utils.utils import parse_numbers


def parse_estimator_id(estimator_id) -> Tuple[GradientEstimator, Dict]:
    """
    Get the Estimator constructor and the estimator configuration from the estimator identifier.
    :param estimator_id: estimator identifier (e.g. `ovis-gamma1`)
    :return: Estimator, config
    """
    if estimator_id == 'Pathwise':
        return Pathwise, {}

    elif estimator_id == 'pathwise-vae':
        return PathwiseVAE, {}

    elif estimator_id == 'pathwise-iwae':
        return PathwiseIWAE, {}

    elif 'wake-wake' == estimator_id:
        return WakeWake, {}

    elif 'wake-sleep' == estimator_id:
        return WakeSleep, {}

    elif 'reinforce' in estimator_id:
        return Reinforce, {}

    elif 'vimco-arithmetic' in estimator_id:
        return VimcoArithmetic, {}

    elif 'vimco-geometric' in estimator_id:
        return VimcoGeometric, {}


    elif 'ovis' in estimator_id:

        if "-S" in estimator_id:  # parse `-S` : number of auxiliary particles
            # the original OVIS-MC used in the paper
            iw_aux = int(eval([s for s in estimator_id.split("-") if 'S' in s][0].replace("S", "")))
            exclusive = "-exclusive" in estimator_id
            Estimator = OvisMonteCarlo
            config = {'iw_aux': iw_aux, 'exclusive': exclusive}

        elif "-arithmetic" in estimator_id or "-geometric" in estimator_id:
            # OVIS~ implementation using `log Z_{[-k]}` given by Vimco.
            assert "-gamma" in estimator_id
            gamma = float(eval([s for s in estimator_id.split("-") if 'gamma' in s][0].replace("gamma", "")))
            Estimator = OvisAsymptoticFromVimco
            config = {'gamma': gamma, 'arithmetic': '-arithmetic' in estimator_id}

        elif "-gamma" in estimator_id:  # parse `-gamma` : parameter of the unified asymptotic approximation
            # the original OVIS~ used in the paper
            gamma = float(eval([s for s in estimator_id.split("-") if 'gamma' in s][0].replace("gamma", "")))
            Estimator = OvisAsymptotic
            config = {'gamma': gamma}

        else:
            raise ValueError(
                f"Ovis estimators should have either of the arguments `-S*` or `-gamma*`. (e.g. `ovis-gamma1`)")

    elif 'tvo' in estimator_id:
        # argument for automatic partition
        if "-config1" in estimator_id:
            partition_id = "config1"
        elif "-config2" in estimator_id:
            partition_id = "config2"
        else:
            partition_id = None

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
        config = {'num_partition': num_partition, 'integration': integration,
                  'partition_id': partition_id, 'log_beta_min': log_beta_min}

    else:
        raise ValueError(f"Unknown estimator {estimator_id}.")

    return Estimator, config
