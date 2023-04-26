import fedml, fedcore
import logging
from fedcore.runner import ExtendedRunner
from fedcore.model.linear.lr import LinearMapping, LogisticRegression
from fedcore.model.cv.resnet import resnet20
from fedcore.model.cv.cnn import CNN_OriginalFedAvg
from fedcore.model.nlp.rnn import RNN_OriginalFedAvg
from fedcore.data.data_loader import load

if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = load(args)

    # load model
    if args.dataset == "mnist":
        model = CNN_OriginalFedAvg()
    elif args.dataset == "cifar10":
        model = resnet20(class_num=output_dim)
    elif args.dataset == "shakespeare":
        model = RNN_OriginalFedAvg()
    elif args.dataset == "synthetic_0_0":
        model = LogisticRegression(60, output_dim)
    elif args.dataset == "synthetic_0.5_0.5":
        model = LogisticRegression(60, output_dim)
    elif args.dataset == "synthetic_1_1":
        model = LogisticRegression(60, output_dim)
    else:
        raise Exception("Exception")

    # start training
    fedml_runner = ExtendedRunner(args, device, dataset, model)
    fedml_runner.run()
