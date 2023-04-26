import torch
from torch import nn
import logging
import copy
from fedml.ml.trainer.my_model_trainer_classification import ModelTrainerCLS
from fedml.core.alg_frame.client_trainer import ClientTrainer


def create_model_trainer(model, args):
    model_trainer = CoresetModelTrainerCLS(model, args)
    return model_trainer


class CoresetModelTrainerCLS(ModelTrainerCLS):
    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        previous_model = copy.deepcopy(model.state_dict())
        # train and update
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        # criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []

            for _, data in enumerate(train_data):
                if len(data) == 3:
                    x, labels, weights = data
                    x, labels, weights = (
                        x.to(device),
                        labels.to(device),
                        weights.to(device),
                    )
                    model.zero_grad()
                    log_probs = model(x)
                    labels = labels.long()
                    loss = torch.div(
                        torch.inner(weights, criterion(log_probs, labels)),
                        torch.sum(weights),
                    )
                    fed_prox_reg = 0.0
                    if args.fedprox_mu != 0.0:
                        for name, param in model.named_parameters():
                            fed_prox_reg += (args.fedprox_mu / 2) * torch.norm(
                                (param - previous_model[name].data.to(device))
                            ) ** 2
                    loss += fed_prox_reg
                else:
                    x, labels = data
                    x, labels = x.to(device), labels.to(device)

                    model.zero_grad()
                    log_probs = model(x)
                    labels = labels.long()
                    loss = torch.mean(criterion(log_probs, labels))
                    fed_prox_reg = 0.0
                    if args.fedprox_mu != 0.0:
                        for name, param in model.named_parameters():
                            fed_prox_reg += (args.fedprox_mu / 2) * torch.norm(
                                (param - previous_model[name].data.to(device))
                            ) ** 2
                    loss += fed_prox_reg

                loss.backward()
                optimizer.step()

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # logging.info(
                #     "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                #         epoch,
                #         (batch_idx + 1) * args.batch_size,
                #         len(train_data) * args.batch_size,
                #         100.0 * (batch_idx + 1) / len(train_data),
                #         loss.item(),
                #     )
                # )

                batch_loss.append(loss.item())
            if len(batch_loss) == 0:
                epoch_loss.append(0.0)
            else:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # logging.info(
            #     "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
            #         self.id, epoch, sum(epoch_loss) / len(epoch_loss)
            #     )
            # )

    def train_init_epoch(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        previous_model = copy.deepcopy(model.state_dict())
        # train and update
        criterion = nn.CrossEntropyLoss(reduction="none").to(
            device
        )  # pylint: disable=E1102
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        epoch_loss = []

        current_epoch = 0
        while current_epoch < 1:
            batch_loss = []

            for _, data in enumerate(train_data):
                if len(data) == 3:
                    x, labels, weights = data
                    x, labels, weights = (
                        x.to(device),
                        labels.to(device),
                        weights.to(device),
                    )
                    model.zero_grad()
                    log_probs = model(x)
                    labels = labels.long()
                    loss = torch.div(
                        torch.inner(weights, criterion(log_probs, labels)),
                        torch.sum(weights),
                    )
                    fed_prox_reg = 0.0
                    if args.fedprox_mu != 0.0:
                        for name, param in model.named_parameters():
                            fed_prox_reg += (args.fedprox_mu / 2) * torch.norm(
                                (param - previous_model[name].data.to(device))
                            ) ** 2
                    loss += fed_prox_reg
                else:
                    x, labels = data
                    x, labels = x.to(device), labels.to(device)

                    model.zero_grad()
                    log_probs = model(x)
                    labels = labels.long()
                    loss = torch.mean(criterion(log_probs, labels))
                    fed_prox_reg = 0.0
                    if args.fedprox_mu != 0.0:
                        for name, param in model.named_parameters():
                            fed_prox_reg += (args.fedprox_mu / 2) * torch.norm(
                                (param - previous_model[name].data.to(device))
                            ) ** 2
                    loss += fed_prox_reg
                loss.backward()

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                # logging.info(
                #     "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                #         epoch,
                #         (batch_idx + 1) * args.batch_size,
                #         len(train_data) * args.batch_size,
                #         100.0 * (batch_idx + 1) / len(train_data),
                #         loss.item(),
                #     )
                # )
                batch_loss.append(loss.item())
            current_epoch += 1
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # logging.info(
            #     "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
            #         self.id, current_epoch, sum(epoch_loss) / len(epoch_loss)
            #     )
            # )

    def train_remaining_epochs(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        previous_model = copy.deepcopy(model.state_dict())
        # train and update
        criterion = nn.CrossEntropyLoss(reduction="none").to(
            device
        )  # pylint: disable=E1102
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        epoch_loss = []

        current_epoch = 1
        while current_epoch < self.args.epochs:
            batch_loss = []

            for _, data in enumerate(train_data):
                if len(data) == 3:
                    x, labels, weights = data
                    x, labels, weights = (
                        x.to(device),
                        labels.to(device),
                        weights.to(device),
                    )
                    model.zero_grad()
                    log_probs = model(x)
                    labels = labels.long()
                    loss = torch.div(
                        torch.inner(weights, criterion(log_probs, labels)),
                        torch.sum(weights),
                    )
                    fed_prox_reg = 0.0
                    if args.fedprox_mu != 0.0:
                        for name, param in model.named_parameters():
                            fed_prox_reg += (args.fedprox_mu / 2) * torch.norm(
                                (param - previous_model[name].data.to(device))
                            ) ** 2
                    loss += fed_prox_reg
                else:
                    x, labels = data
                    x, labels = x.to(device), labels.to(device)

                    model.zero_grad()
                    log_probs = model(x)
                    labels = labels.long()
                    loss = torch.mean(criterion(log_probs, labels))
                    fed_prox_reg = 0.0
                    if args.fedprox_mu != 0.0:
                        for name, param in model.named_parameters():
                            fed_prox_reg += (args.fedprox_mu / 2) * torch.norm(
                                (param - previous_model[name].data.to(device))
                            ) ** 2
                    loss += fed_prox_reg
                loss.backward()

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                # logging.info(
                #     "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                #         epoch,
                #         (batch_idx + 1) * args.batch_size,
                #         len(train_data) * args.batch_size,
                #         100.0 * (batch_idx + 1) / len(train_data),
                #         loss.item(),
                #     )
                # )
                batch_loss.append(loss.item())

            current_epoch += 1
            if len(batch_loss) == 0:
                continue
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # logging.info(
            #     "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
            #         self.id, current_epoch, sum(epoch_loss) / len(epoch_loss)
            #     )
            # )

    def train_num_epochs(self, train_data, device, args, epochs_num=None):
        model = self.model

        model.to(device)
        model.train()

        previous_model = copy.deepcopy(model.state_dict())
        # train and update
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        # criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        epoch_loss = []
        epochs_num = args.epochs if not epochs_num else epochs_num
        for epoch in range(epochs_num):
            batch_loss = []

            for _, data in enumerate(train_data):
                if len(data) == 3:
                    x, labels, weights = data
                    x, labels, weights = (
                        x.to(device),
                        labels.to(device),
                        weights.to(device),
                    )
                    model.zero_grad()
                    log_probs = model(x)
                    labels = labels.long()
                    loss = torch.div(
                        torch.inner(weights, criterion(log_probs, labels)),
                        torch.sum(weights),
                    )
                    fed_prox_reg = 0.0
                    if args.fedprox_mu != 0.0:
                        for name, param in model.named_parameters():
                            fed_prox_reg += (args.fedprox_mu / 2) * torch.norm(
                                (param - previous_model[name].data.to(device))
                            ) ** 2
                    loss += fed_prox_reg
                else:
                    x, labels = data
                    x, labels = x.to(device), labels.to(device)

                    model.zero_grad()
                    log_probs = model(x)
                    labels = labels.long()
                    loss = torch.mean(criterion(log_probs, labels))
                    fed_prox_reg = 0.0
                    if args.fedprox_mu != 0.0:
                        for name, param in model.named_parameters():
                            fed_prox_reg += (args.fedprox_mu / 2) * torch.norm(
                                (param - previous_model[name].data.to(device))
                            ) ** 2
                    loss += fed_prox_reg

                loss.backward()
                optimizer.step()

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # logging.info(
                #     "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                #         epoch,
                #         (batch_idx + 1) * args.batch_size,
                #         len(train_data) * args.batch_size,
                #         100.0 * (batch_idx + 1) / len(train_data),
                #         loss.item(),
                #     )
                # )

                batch_loss.append(loss.item())
            if len(batch_loss) == 0:
                epoch_loss.append(0.0)
            else:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # logging.info(
            #     "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
            #         self.id, epoch, sum(epoch_loss) / len(epoch_loss)
            #     )
            # )

        def test(self, test_data, device, args):
            model = self.model

            model.to(device)
            model.eval()

            metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}

            criterion = nn.CrossEntropyLoss().to(device)

            with torch.no_grad():
                for batch_idx, (x, target) in enumerate(test_data):
                    x = x.to(device)
                    target = target.to(device)
                    pred = model(x)
                    target = target.long()
                    loss = criterion(pred, target)  # pylint: disable=E1102

                    _, predicted = torch.max(pred, -1)
                    correct = predicted.eq(target).sum()

                    metrics["test_correct"] += correct.item()
                    metrics["test_loss"] += loss.item() * target.size(0)
                    metrics["test_total"] += target.size(0)
            return metrics


class FedProxModelTrainer(ClientTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        previous_model = copy.deepcopy(model.state_dict())

        # train and update
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, data in enumerate(train_data):
                if len(data) == 3:
                    x, labels, weights = data
                    x, labels, weights = (
                        x.to(device),
                        labels.to(device),
                        weights.to(device),
                    )
                    model.zero_grad()
                    log_probs = model(x)
                    loss = torch.div(
                        torch.inner(weights, criterion(log_probs, labels)),
                        torch.sum(weights),
                    )  # pylint: disable=E1102
                    # if args.fedprox:
                    fed_prox_reg = 0.0
                    for name, param in model.named_parameters():
                        fed_prox_reg += (args.fedprox_mu / 2) * torch.norm(
                            (param - previous_model[name].data.to(device))
                        ) ** 2
                    loss += fed_prox_reg
                else:
                    x, labels = data
                    x, labels = x.to(device), labels.to(device)

                    model.zero_grad()
                    log_probs = model(x)
                    loss = torch.mean(
                        criterion(log_probs, labels)
                    )  # pylint: disable=E1102
                    # if args.fedprox:
                    fed_prox_reg = 0.0
                    for name, param in model.named_parameters():
                        fed_prox_reg += (args.fedprox_mu / 2) * torch.norm(
                            (param - previous_model[name].data.to(device))
                        ) ** 2
                    loss += fed_prox_reg

                loss.backward()

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                # logging.info(
                #     "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                #         epoch,
                #         (batch_idx + 1) * args.batch_size,
                #         len(train_data) * args.batch_size,
                #         100.0 * (batch_idx + 1) / len(train_data),
                #         loss.item(),
                #     )
                # )
                batch_loss.append(loss.item())
            if len(batch_loss) == 0:
                epoch_loss.append(0.0)
            else:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # logging.info(
            #     "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
            #         self.id, epoch, sum(epoch_loss) / len(epoch_loss)
            #     )
            # )

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)  # pylint: disable=E1102

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)
        return metrics

    def train_num_epochs(self, train_data, device, args, epochs_num=None):
        model = self.model

        model.to(device)
        model.train()

        previous_model = copy.deepcopy(model.state_dict())

        # train and update
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        epoch_loss = []
        epochs_num = args.epochs if not epochs_num else epochs_num
        for epoch in range(epochs_num):
            batch_loss = []

            for batch_idx, data in enumerate(train_data):
                if len(data) == 3:
                    x, labels, weights = data
                    x, labels, weights = (
                        x.to(device),
                        labels.to(device),
                        weights.to(device),
                    )
                    model.zero_grad()
                    log_probs = model(x)
                    loss = torch.div(
                        torch.inner(weights, criterion(log_probs, labels)),
                        torch.sum(weights),
                    )  # pylint: disable=E1102
                    # if args.fedprox:
                    fed_prox_reg = 0.0
                    for name, param in model.named_parameters():
                        fed_prox_reg += (args.fedprox_mu / 2) * torch.norm(
                            (param - previous_model[name].data.to(device))
                        ) ** 2
                    loss += fed_prox_reg
                else:
                    x, labels = data
                    x, labels = x.to(device), labels.to(device)

                    model.zero_grad()
                    log_probs = model(x)
                    loss = torch.mean(
                        criterion(log_probs, labels)
                    )  # pylint: disable=E1102
                    # if args.fedprox:
                    fed_prox_reg = 0.0
                    for name, param in model.named_parameters():
                        fed_prox_reg += (args.fedprox_mu / 2) * torch.norm(
                            (param - previous_model[name].data.to(device))
                        ) ** 2
                    loss += fed_prox_reg

                loss.backward()

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                # logging.info(
                #     "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                #         epoch,
                #         (batch_idx + 1) * args.batch_size,
                #         len(train_data) * args.batch_size,
                #         100.0 * (batch_idx + 1) / len(train_data),
                #         loss.item(),
                #     )
                # )
                batch_loss.append(loss.item())
            if len(batch_loss) == 0:
                epoch_loss.append(0.0)
            else:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # logging.info(
            #     "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
            #         self.id, epoch, sum(epoch_loss) / len(epoch_loss)
            #     )
            # )
