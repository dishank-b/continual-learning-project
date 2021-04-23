from copy import deepcopy
import torch
from continuum.tasks import split_train_val
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np

# TODO use Sequoia
# FIXME continual trainer not found file bug due to not saving flag maybe ??
# FIXME Index out of bound in sample mean and variance estimation giving label 3
class CustomDataset:
    def __init__(self, dataset, normalize_classes=False, n_prev_classes=0) -> None:
        self.dataset = dataset
        self.normalize_classes = normalize_classes
        self.n_prev_classes = n_prev_classes

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return self

    def __getitem__(self, inx):
        x, y, t = self.dataset[inx]
        if self.normalize_classes:
            y = y - self.n_prev_classes
        return x, y


class ContinualTrainer:
    def __init__(
        self,
        scenario,
        test_dataset,
        model,
        start_lr,
        batch_size,
        mahalanobis,
        in_transform,
        epochs,
    ) -> None:
        self.scenario = scenario
        self.test_loader = DataLoader(CustomDataset(test_dataset), batch_size, False)
        self.model = model
        self.mahalanobis = mahalanobis
        self.start_lr = start_lr
        self.batch_size = batch_size
        self.loss_fn = nn.CrossEntropyLoss()
        # regularizer to keep features in the same space from previous task
        self.network_regularizer = 0.1
        self.regularizer_loss = nn.MSELoss()
        self.in_transform = in_transform
        decay = 0.1
        self.lr_lambda = (
            lambda epoch: decay
            if epoch > 0 and (epochs / epoch == 2 or epoch / epochs == 0.75)
            else 1
        )
        self.epochs = epochs
        self.device = self.mahalanobis.device

    def configure_optimizers(self, model):
        optimizer = torch.optim.SGD(model.parameters(), lr=self.start_lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self.lr_lambda)
        return optimizer, scheduler

    def fit(self):
        prev_model = None
        model = deepcopy(self.model)
        optimizer, scheduler = self.configure_optimizers(model)
        total_mean, total_precision = None, None
        n_prev_classes = 0
        for task_id, train_taskset in enumerate(self.scenario):
            # logging
            print("Starting training of task {}".format(task_id))
            train_taskset, val_taskset = split_train_val(train_taskset, val_split=0.1)
            train_loader = DataLoader(
                CustomDataset(train_taskset), batch_size=self.batch_size, shuffle=True
            )
            val_loader = DataLoader(
                CustomDataset(val_taskset), batch_size=self.batch_size, shuffle=True
            )
            n_classes = len(train_taskset.get_classes())
            for _ in range(self.epochs):
                for x, y in train_loader:
                    optimizer.zero_grad()
                    x = x.to(self.device)
                    y = y.to(self.device)
                    y_hat = model(x)
                    loss = self.loss_fn(y_hat, y)
                    if prev_model is not None:
                        # TODO switch between fine tune and regularization based on OOD score
                        current_logit = model.penultimate_forward(x)[-1]
                        prev_logit = prev_model.penultimate_forward(x)[-1].detach()
                        loss += self.network_regularizer * self.regularizer_loss(
                            current_logit, prev_logit
                        )
                    loss.backward()
                    optimizer.step()
                    scheduler.step() 
            # save current task network
            prev_model = deepcopy(model)
            self.mahalanobis.update_network(model)
            # computing mean and precision
            if n_prev_classes == 0:
                train_loader_partial = train_loader
                val_loader_partial = val_loader
            else:
                train_loader_partial = DataLoader(
                    CustomDataset(
                        train_taskset,
                        normalize_classes=True,
                        n_prev_classes=n_prev_classes,
                    ),
                    batch_size=self.batch_size,
                    shuffle=True,
                )
                val_loader_partial = DataLoader(
                    CustomDataset(
                        val_taskset,
                        normalize_classes=True,
                        n_prev_classes=n_prev_classes,
                    ),
                    batch_size=self.batch_size,
                    shuffle=True,
                )
            n_prev_classes += n_classes
            sample_mean, precision = self.mahalanobis.sample_estimator(
                train_loader_partial, n_classes
            )
            if total_mean is None:
                total_mean = sample_mean
                total_precision = precision
            else:
                for i in range(len(precision)):
                    total_precision[i] = (n_classes / (n_classes + 1)) * total_precision[i] + (
                        1 / (n_classes + 1)
                    ) * precision[i]
                total_mean = total_mean + sample_mean
            self.mahalanobis.sample_mean = sample_mean
            self.mahalanobis.precision = precision
            # check OOD performance
            self.mahalanobis.compute_all_noise_mahalanobis(
                val_loader_partial, self.in_transform, n_classes, m_list=[0.001]
            )
            self.mahalanobis.cross_validate(m_list=[0.001])
        # TODO compute test set performance final

        # compute final OOD performance
        self.mahalanobis.sample_mean = total_mean
        self.mahalanobis.precision = total_precision
        self.mahalanobis.compute_all_noise_mahalanobis(
            self.test_loader,
            self.in_transform,
            self.mahalanobis.args.num_classes,
            m_list=[0.001],
        )
        self.mahalanobis.cross_validate(m_list=[0.001])
