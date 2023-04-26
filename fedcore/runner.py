from torch import nn

from fedml.constants import FEDML_SIMULATION_TYPE_SP
from fedml import FedMLRunner
from .fedcore_api import FedCoreAPI
from .fedavg_api import FedAvgAPI
from .fedprox_api import FedProxAPI
from .constants import (
    FedML_FEDERATED_OPTIMIZER_FEDCORE,
    FedML_FEDERATED_OPTIMIZER_FEDAVG,
    FedML_FEDERATED_OPTIMIZER_FEDPROX,
)

SUPPORTED_ALGOS = [
    FedML_FEDERATED_OPTIMIZER_FEDCORE,
    FedML_FEDERATED_OPTIMIZER_FEDAVG,
    FedML_FEDERATED_OPTIMIZER_FEDPROX,
]


class ExtendedRunner(FedMLRunner):
    def _init_simulation_runner(
        self, args, device, dataset, model, client_trainer=None, server_aggregator=None
    ):

        if (
            hasattr(args, "backend")
            and args.backend == FEDML_SIMULATION_TYPE_SP
            and args.federated_optimizer in SUPPORTED_ALGOS
        ):
            runner = SupportedSimulatorSP(
                args, device, dataset, model, client_trainer, server_aggregator
            )
        else:
            runner = super()._init_simulation_runner(
                args, device, dataset, model, client_trainer, server_aggregator
            )

        return runner


class SupportedSimulatorSP:
    def __init__(
        self, args, device, dataset, model, client_trainer=None, server_aggregator=None
    ):
        if args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDCORE:
            self.fl_trainer = FedCoreAPI(args, device, dataset, model)
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDAVG:
            self.fl_trainer = FedAvgAPI(args, device, dataset, model)
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDPROX:
            self.fl_trainer = FedProxAPI(args, device, dataset, model)
        else:
            raise Exception("Exception")

    def run(self):
        self.fl_trainer.train()
