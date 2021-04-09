from copy import deepcopy
import torch
from continuum.tasks import split_train_val
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np


class ContinualTrainer:
    def __init__(
        self,
        scenario,
        test_loader,
        model,
        start_lr,
        batch_size,
        mahalanobis,
        in_transform,
    ) -> None:
        self.scenario = scenario
        self.test_loader = test_loader
        self.model = model
        self.mahalanobis = mahalanobis
        self.start_lr = start_lr
        self.batch_size = batch_size
        self.loss_fn = nn.CrossEntropyLoss()
        # regularizer to keep features in the same space from previous task
        self.network_regularizer = 0.1
        self.regularizer_loss = nn.MSELoss()
        self.in_transform = in_transform

    def configure_optimizers(self, model):
        optimizer = torch.optim.SGD(model.parameters(), lr=self.start_lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self.lr_lambda)
        return optimizer, scheduler

    def fit(self):
        prev_model = None
        model = deepcopy(self.model)
        optimizer, scheduler = self.configure_optimizers(model)
        total_mean, total_precision = None, None
        for task_id, train_taskset in enumerate(self.scenario):
            # logging
            print("Starting training of task {}".format(task_id))
            train_taskset, val_taskset = split_train_val(train_taskset, val_split=0.1)
            train_loader = DataLoader(
                train_taskset, batch_size=self.batch_size, shuffle=True
            )
            val_loader = DataLoader(
                val_taskset, batch_size=self.batch_size, shuffle=True
            )
            n_classes = len(train_taskset.get_classes())
            for x, y, t in train_loader:
                optimizer.zero_grad()
                y_hat = model(x)
                loss = self.loss_fn(y_hat, y)
                if prev_model is not None:
                    # TODO switch between fine tune and regularization based on OOD score
                    current_logit = model.penultimate_forward(x)
                    prev_logit = prev_model.penultimate_forward(x)
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
            sample_mean, precision = self.mahalanobis.sample_estimator(
                train_loader, n_classes
            )
            if total_mean is None:
                total_mean = sample_mean
                total_precision = precision
            else:
                total_precision = (n_classes / (n_classes + 1)) * total_precision + (
                    1 / (n_classes + 1)
                ) * precision
                total_mean = total_mean + sample_mean
            self.mahalanobis.sample_mean = sample_mean
            self.mahalanobis.precision = precision
            # check OOD performance
            self.mahalanobis.compute_all_noise_mahalanobis(
                val_loader, self.in_transform, m_list=[0.001]
            )
            self.mahalanobis.cross_validate(m_list=[0.001])
        # TODO compute test set performance final 

        # compute final OOD performance
        self.mahalanobis.sample_mean = total_mean
        self.mahalanobis.precision = total_precision
        self.mahalanobis.compute_all_noise_mahalanobis(
            self.test_loader, self.in_transform, m_list=[0.001]
        )
        self.mahalanobis.cross_validate(m_list=[0.001])