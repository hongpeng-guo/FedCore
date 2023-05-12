# FedCore: Straggler-Free Federated Learning withDistributed Core-sets

## Installation

```Python
pip install -r requirements.txt
```

## Run an experiment on MINIST dataset with FedCore

```Python
python main.py --cf config/mnist_fedcore.yaml
```

## Run all experiments as shown in the paper

```Python
./run_all.sh
```

## More self-defined simulation

You can create self-defined simulation scenarios by modifying or creating new `yaml` configuration files as presented in `/configs/*.ymal`.
