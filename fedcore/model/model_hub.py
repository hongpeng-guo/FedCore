import logging
import torch.nn as nn
import fedml
from .cv.cnn import CNN_DropOut, CNN_WEB
from .cv.efficientnet import EfficientNet
from .cv.mobilenet import mobilenet
from .cv.resnet import resnet56
from .linear.lr import LogisticRegression
from .nlp.rnn import RNN_OriginalFedAvg

from ..constants import FedML_FEDERATED_OPTIMIZER_FEDCORE


def create(args, output_dim):
    global model
    model_name = args.model
    logging.info(
        "create_model. model_name = %s, output_dim = %s" % (model_name, output_dim)
    )
    if args.federated_optimizer != FedML_FEDERATED_OPTIMIZER_FEDCORE:
        model = fedml.model.create(args, output_dim)
    elif model_name == "lr" and args.dataset == "mnist":
        logging.info("LogisticRegression + MNIST")
        model = LogisticRegression(28 * 28, output_dim)
    elif model_name == "cnn_web" and args.dataset == "cifar10":
        logging.info("CNN_WEB + CIFAR10")
        model = CNN_WEB()
    elif model_name == "cnn" and args.dataset == "mnist":
        logging.info("CNN + MNIST")
        model = CNN_DropOut(False)
    elif model_name == "cnn" and args.dataset == "femnist":
        logging.info("CNN + FederatedEMNIST")
        model = CNN_DropOut(False)
    elif model_name == "rnn" and args.dataset == "shakespeare":
        logging.info("RNN + shakespeare")
        model = RNN_OriginalFedAvg()
    elif model_name == "lr" and args.dataset == "stackoverflow_lr":
        logging.info("lr + stackoverflow_lr")
        model = LogisticRegression(10000, output_dim)
    elif model_name == "resnet56":
        model = resnet56(class_num=output_dim)
    elif model_name == "mobilenet":
        model = mobilenet(class_num=output_dim)
    elif model_name == "efficientnet":
        model = EfficientNet()
    else:
        raise Exception(
            "no such model definition, please check the argument spelling or customize your own model"
        )
    return model
