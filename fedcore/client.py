import torch
from torch import nn
import math
import numpy as np
import logging
from torch.utils.data import TensorDataset, DataLoader, Subset, Dataset
from fedml.ml.engine.ml_engine_adapter import convert_numpy_to_ml_engine_data_format
from sklearn.metrics.pairwise import euclidean_distances
from kmedoids import fasterpam


class Client:
    def __init__(
        self,
        client_idx,
        local_training_data,
        local_test_data,
        local_sample_number,
        compute_power,
        args,
        device,
        model_trainer,
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.compute_power = compute_power

        self.args = args
        self.device = device
        self.model_trainer = model_trainer

    def update_local_dataset(
        self,
        client_idx,
        local_training_data,
        local_test_data,
        local_sample_number,
        compute_power,
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.compute_power = compute_power
        self.model_trainer.set_id(client_idx)

    def get_sample_number(self):
        return self.local_sample_number

    def update_training_coreset(self, epoch_ddl):
        coreset_size = int(math.ceil(self.compute_power * epoch_ddl))
        fullset_size = len(self.local_training_data) * self.args.batch_size
        # logging.info("true_set:{}".format(min(coreset_size, fullset_size)))
        if coreset_size >= fullset_size:
            return
        if self.args.dataset == "cifar10":
            self.local_training_data = self.generate_coreset_w_augment(epoch_ddl)
        else:
            self.local_training_data = self.generate_coreset(epoch_ddl)

    def train(self, w_global):
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        return weights

    def train_init_epoch(self, w_global):
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train_init_epoch(
            self.local_training_data, self.device, self.args
        )
        weights = self.model_trainer.get_model_params()
        return weights

    def train_remaining_epochs(self, w_global):
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train_remaining_epochs(
            self.local_training_data, self.device, self.args
        )
        weights = self.model_trainer.get_model_params()
        return weights

    def train_num_epochs(self, w_global, epoch_num=None):
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train_num_epochs(
            self.local_training_data, self.device, self.args, epoch_num
        )
        weights = self.model_trainer.get_model_params()
        return weights

    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics

    def coreset_grads(self, train_data):
        device = self.device
        args = self.args
        model = self.model_trainer.model

        gradients = []
        if args.model == "lr":
            for _, (x, target) in enumerate(train_data):
                gradients.append(x.flatten(1).cpu().numpy())
            return np.concatenate(gradients, axis=0)

        model.to(device)
        model.eval()
        criterion = nn.CrossEntropyLoss().to(device)

        for _, (x, target) in enumerate(train_data):
            x = x.to(device)
            target = target.to(device)
            with model.embedding_recorder:
                pred = model(x)
            loss = criterion(pred, target)  # pylint: disable=E1102

            with torch.no_grad():
                batch_num = target.shape[0]
                classes_num = pred.shape[1]
                embedding = model.embedding_recorder.embedding
                embedding_dim = model.get_last_layer().in_features
                bias_grads = torch.autograd.grad(loss, pred)[0]
                weights_grads = embedding.view(batch_num, 1, embedding_dim).repeat(
                    1, classes_num, 1
                )
                weights_grads *= bias_grads.view(batch_num, classes_num, 1).repeat(
                    1, 1, embedding_dim
                )
                tmp_grad = (
                    torch.cat([bias_grads, weights_grads.flatten(1)], dim=1)
                    .cpu()
                    .numpy()
                )
                gradients.append(tmp_grad)

        return np.concatenate(gradients, axis=0)

    def generate_coreset(self, epoch_ddl):
        # fix train_data as a list rather than a random dataloader.
        train_data = [(x, y) for (x, y) in self.local_training_data]
        model = self.model_trainer.model

        train_x_np = np.concatenate([x.numpy() for (x, _) in train_data], axis=0)
        train_y_np = np.concatenate([y.numpy() for (_, y) in train_data], axis=0)

        gradients = self.coreset_grads(train_data)
        num_classes = model.get_last_layer().out_features
        num_train_data = gradients.shape[0]
        coreset_size = int(
            math.ceil(
                (self.compute_power * epoch_ddl * self.args.epochs - num_train_data)
                / (self.args.epochs - 1)
            )
        )
        coreset_size = max(0, coreset_size)
        data_x, data_y, data_weight = [], [], []
        included_idx = list()
        for idx in range(num_classes):
            class_idx = np.argwhere(train_y_np == idx).reshape(-1)
            patial_grads = gradients[class_idx]
            class_total_size = patial_grads.shape[0]
            class_coreset_size = int(
                math.ceil(coreset_size * class_total_size / num_train_data)
            )
            class_coreset_size = min(class_coreset_size, class_total_size)
            if class_coreset_size <= 0:
                continue
            # logging.info(
            #     "class_total_size: {}, expected_class_coreset_size: {}".format(
            #         class_total_size, class_coreset_size
            #     )
            # )
            diss = euclidean_distances(patial_grads)
            res = fasterpam(diss, class_coreset_size)
            labels, medoids = res.labels, res.medoids

            for core in medoids:
                included_idx.append(class_idx[core])
                data_x.append(train_x_np[class_idx[core]])
                data_y.append(train_y_np[class_idx[core]])
                data_weight.append(np.count_nonzero(labels == labels[core]))

        # print(
        #     "include all sample: {}".format(
        #         list(range(num_train_data)) == sorted(included_idx)
        #     )
        # )
        # generate TensorDataset for training data
        train_data_new = TensorDataset(
            *self.convert_list_to_torch_data_format(data_x, data_y, data_weight)
        )
        if len(train_data_new) <= 0:
            return []
        train_loader = DataLoader(
            train_data_new, batch_size=self.args.batch_size, shuffle=True
        )
        return train_loader

    def convert_list_to_torch_data_format(self, x, y, w):
        import torch
        import numpy as np

        if self.args.model == "cnn" and self.args.dataset == "mnist":
            data_x = (
                torch.from_numpy(np.asarray(x)).float().reshape(-1, 28, 28)
            )  # CNN_MINST
        elif self.args.model == "rnn":
            data_x = torch.from_numpy(np.asarray(x)).long()
        else:
            data_x = torch.from_numpy(np.asarray(x)).float()  # LR_MINST or other
        data_y = torch.from_numpy(np.asarray(y)).long()
        data_weight = torch.from_numpy(np.asarray(w)).float()
        return data_x, data_y, data_weight

    def generate_coreset_w_augment(self, epoch_ddl):
        # fix train_data as a list rather than a random dataloader.
        train_data = DataLoader(
            self.local_training_data.dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
        )
        model = self.model_trainer.model

        gradients = self.coreset_grads(train_data)
        train_y_np = np.concatenate([y.numpy() for (_, y) in train_data], axis=0)

        num_classes = model.get_last_layer().out_features
        num_train_data = gradients.shape[0]
        coreset_size = int(
            math.ceil(
                (self.compute_power * epoch_ddl * self.args.epochs - num_train_data)
                / (self.args.epochs - 1)
            )
        )
        coreset_size = max(0, coreset_size)
        coreset_idx, coreset_weight = [], []
        for idx in range(num_classes):
            class_idx = np.argwhere(train_y_np == idx).reshape(-1)
            patial_grads = gradients[class_idx]
            class_total_size = patial_grads.shape[0]
            class_coreset_size = int(
                math.ceil(coreset_size * class_total_size / num_train_data)
            )
            class_coreset_size = min(class_coreset_size, class_total_size)
            if class_coreset_size <= 0:
                continue
            diss = euclidean_distances(patial_grads)
            res = fasterpam(diss, class_coreset_size)
            labels, medoids = res.labels, res.medoids

            for core in medoids:
                coreset_idx.append(class_idx[core])
                coreset_weight.append(np.count_nonzero(labels == labels[core]))

        coreset_weight = torch.from_numpy(np.asarray(coreset_weight)).float()
        train_coreset = WeightedSubset(train_data.dataset, coreset_idx, coreset_weight)
        # generate TensorDataset for training data
        train_loader = DataLoader(
            train_coreset, batch_size=self.args.batch_size, shuffle=True
        )
        return train_loader


class WeightedSubset(Dataset):
    def __init__(self, dataset, indices, weights) -> None:
        self.dataset = dataset
        self.indices = indices
        self.weights = weights

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] + (self.weights[i],) for i in idx]]
        return self.dataset[self.indices[idx]] + (self.weights[idx],)

    def __len__(self):
        return len(self.indices)
