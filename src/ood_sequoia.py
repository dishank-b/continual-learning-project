from sequoia.methods import Method
from sequoia.settings import ClassIncrementalSetting
from sequoia.settings.passive import PassiveEnvironment
from sequoia.settings.passive.cl.objects import (
    Actions,
    Environment,
    Observations,
    PassiveEnvironment,
    Rewards,
)
from dataclasses import dataclass

# Hparams include all hyperparameters for all methods
from simple_parsing import ArgumentParser
from typing import Dict, Optional, Tuple
from sequoia.common.hparams import HyperParameters
import torch
from torch import Tensor
from numpy import inf, floor
import tqdm
import gym
import wandb
from torch.utils.data import DataLoader
import torch.nn as nn
from copy import deepcopy


class CustomDataset:
    """Custom Dataset used to remove task t from the batch to use deep_mahalanobis_detector library
    """

    def __init__(self, dataset, normalize_classes=False, n_prev_classes=0) -> None:
        """Initialize Custom dataset

        Args:
            dataset (torch.dataset): pytorch dataset
            normalize_classes (bool, optional): a flag to normalize classes to start from 0. Defaults to False.
            n_prev_classes (int, optional): number of prev classes to normalize output. Defaults to 0.
        """
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


class OODSequoia(Method, target_setting=ClassIncrementalSetting):
    """EWC implementation based on Sequoia library examples
    """

    @dataclass
    class HParams(HyperParameters):
        start_lr: float = 3e-4
        batch_size: int = 32
        epochs: int = 1
        early_stop_patience: int = 1
        wandb_logging: bool = False
        # regularizer to keep features in the same space from previous task
        ood_regularizer: float = 0.5
        lwf_regularizer: float = 0
        temperature_lwf: float = 2
        # decay of max_epochs after each task
        epochs_decay_per_task: float = 0.95
        noise_ood: float = 0.001
        # compute risk associated with each task to enable/disable lwf like loss with ood_regularizer/2
        compute_risk: bool = True

    def __init__(
        self, test_loader, model, mahalanobis, in_transform, hparams=None,
    ) -> None:
        """Initialize OOD Sequoia (Our approach)

        Args:
            test_dataset (torch.dataloader): dataset used for testing purposes
            model (nn.Module): pytorch Model
            mahalanobis (MahalanobisCompute): object used to compute mahalanobis distance
            in_transform (list): list of transformation applied to input
            hparams (HParams, optional): hyperparameters specific to the method. Defaults to None.
        """
        self.hparams = hparams or OODSequoia.HParams()
        self.model = model
        self.mahalanobis = mahalanobis
        self.optimizer: torch.optim.Optimizer
        self.in_transform = in_transform
        self.test_loader = test_loader
        self.scheduler = None
        self.n_classes = None
        self.n_seen_classes = None
        self.prev_model = None
        self.total_mean = None
        self.total_precision = None
        self.ood_results = None
        self.total_n_tasks = None
        decay = 0.1
        self.lr_lambda = (
            lambda epoch: decay
            if epoch > 0
            and (
                self.hparams.epochs / epoch == 2 or epoch / self.hparams.epochs == 0.75
            )
            else 1
        )
        self.m_list = [self.hparams.noise_ood]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def configure(self, setting: ClassIncrementalSetting):
        """Called before the method is applied on a setting (before training).

        You can use this to instantiate your model, for instance, since this is
        where you get access to the observation & action spaces.

        Args:
            setting (ClassIncrementalSetting): Current Continual training setting
        """
        self.total_n_tasks = setting.nb_tasks
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.hparams.start_lr, momentum=0.9
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, self.lr_lambda
        )
        self.task = 0
        self.n_examples_seen = 0
        self.n_seen_classes = 0
        self.prev_model = None
        self.total_mean = None
        self.total_precision = None
        self.ood_results = None
        self.task_risk = None

    def fit(self, train_env: PassiveEnvironment, valid_env: PassiveEnvironment):
        """Example train loop.
        You can do whatever you want with train_env and valid_env here.

        NOTE: In the Settings where task boundaries are known (in this case all
        the supervised CL settings), this will be called once per task.

        Args:
            train_env (PassiveEnvironment): current task train environment
            valid_env (PassiveEnvironment): current task validation environment
        """
        best_model = self.model.state_dict()
        if self.scheduler is not None:
            best_scheduler = self.scheduler.state_dict()
        best_optimizer = self.optimizer.state_dict()

        print(f"task {self.task}")
        best_val_loss = inf
        best_epoch = 0
        self.memorize = False
        self.n_classes = train_env.dataset.nb_classes
        if self.hparams.compute_risk and self.task > 0:
            self.task_risk = self.compute_risk(train_env)
        for epoch in range(int(floor(self.hparams.epochs))):
            self.model.train()
            print(f"Starting epoch {epoch}")
            # Training loop:
            torch.set_grad_enabled(True)

            with tqdm.tqdm(train_env) as train_pbar:
                postfix = {}
                train_pbar.set_description(f"Training Epoch {epoch}")
                for i, batch in enumerate(train_pbar):
                    loss, metrics_dict = self.shared_step(
                        batch, environment=train_env, validation=False, epoch=epoch,
                    )
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    postfix.update(metrics_dict)
                    train_pbar.set_postfix(postfix)
                    self.n_examples_seen += len(batch[0].x)
                    if self.hparams.wandb_logging:
                        wandb.log(metrics_dict)
            if self.scheduler is not None:
                self.scheduler.step()

            # Validation loop:
            self.model.eval()
            torch.set_grad_enabled(False)
            with tqdm.tqdm(valid_env) as val_pbar:
                postfix = {}
                val_pbar.set_description(f"Validation Epoch {epoch}")
                epoch_val_loss = 0.0

                for i, batch in enumerate(val_pbar):
                    batch_val_loss, metrics_dict = self.shared_step(
                        batch, environment=valid_env, validation=True, epoch=epoch,
                    )
                    epoch_val_loss += batch_val_loss
                    postfix.update(metrics_dict, val_loss=epoch_val_loss.item())
                    val_pbar.set_postfix(postfix)

                if self.hparams.wandb_logging:
                    wandb.log({"val_loss": epoch_val_loss})

            if epoch_val_loss <= best_val_loss and epoch > 0:
                best_val_loss = epoch_val_loss
                best_epoch = epoch
                best_model = self.model.state_dict()
                if self.scheduler is not None:
                    best_scheduler = self.scheduler.state_dict()
                best_optimizer = self.optimizer.state_dict()

            if epoch - best_epoch > self.hparams.early_stop_patience:
                print(f"Early stopping at epoch {epoch}. ")
                break

        if best_val_loss != inf:
            if self.hparams.wandb_logging:
                wandb.run.summary[f"best epoch task {self.task}"] = best_epoch
            print(f"Loading model from epoch {best_epoch}!")
            self.model.load_state_dict(best_model)
            if self.scheduler is not None:
                self.scheduler.load_state_dict(best_scheduler)
            self.optimizer.load_state_dict(best_optimizer)
        elif self.hparams.wandb_logging:
            wandb.run.summary[f"best epoch task {self.task}"] = self.hparams.epochs
        self.model.train()
        # Training loop:
        torch.set_grad_enabled(True)
        if self.mahalanobis is not None:
            self._mahalanobis_update(train_env, valid_env)

        self.n_seen_classes += self.n_classes
        self.hparams.epochs *= self.hparams.epochs_decay_per_task
        if self.task == self.total_n_tasks - 1:
            # last task fitting done
            self.compute_final()
        self.task += 1
        self.prev_model = deepcopy(self.model)

    def _mahalanobis_update(
        self, train_env: PassiveEnvironment, valid_env: PassiveEnvironment
    ):
        """Updating mahalanobis distance values after each task

        Args:
            train_env (PassiveEnvironment): Training environment
            valid_env (PassiveEnvironment): Validation environment
        """
        self.mahalanobis.update_network(self.model)
        train_loader_partial = DataLoader(
            CustomDataset(
                train_env.dataset,
                normalize_classes=self.n_seen_classes > 0,
                n_prev_classes=self.n_seen_classes,
            ),
            batch_size=32,
            shuffle=True,
        )
        val_loader_partial = DataLoader(
            CustomDataset(
                valid_env.dataset,
                normalize_classes=self.n_seen_classes > 0,
                n_prev_classes=self.n_seen_classes,
            ),
            batch_size=32,
            shuffle=True,
        )
        n_classes = train_env.dataset.nb_classes
        sample_mean, precision = self.mahalanobis.sample_estimator(
            train_loader_partial, n_classes
        )
        if self.total_mean is None:
            self.total_mean = sample_mean
            self.total_precision = precision
        else:
            for i in range(len(precision)):
                self.total_precision[i] = (
                    n_classes / (n_classes + 1)
                ) * self.total_precision[i] + (1 / (n_classes + 1)) * precision[i]
            for i in range(len(sample_mean)):
                self.total_mean[i] = torch.cat((self.total_mean[i], sample_mean[i]), 0)
        self.mahalanobis.sample_mean = sample_mean
        self.mahalanobis.precision = precision
        self.mahalanobis.compute_all_noise_mahalanobis(
            val_loader_partial, self.in_transform, n_classes, m_list=self.m_list
        )
        self.ood_results = self.mahalanobis.cross_validate(
            m_list=self.m_list, save_classifier=self.hparams.compute_risk
        )

    def compute_risk(self, train_env: PassiveEnvironment):
        """Computes risk associated with a task

        Args:
            train_env (PassiveEnvironment): input train environment

        Returns:
            float: risk value 1 means high risk ood and 0 no risk
        """
        if self.mahalanobis is None:
            # 1 means high risk ood and 0 no risk
            return 0
        # FIXME a bug in risk computation
        train_loader_partial = DataLoader(
            CustomDataset(train_env.dataset, normalize_classes=False,),
            batch_size=32,
            shuffle=True,
        )
        risk = self.mahalanobis.compute_task_risk(
            train_loader_partial, self.m_list[0], self.n_seen_classes
        )
        return risk

    def get_actions(
        self, observations: Observations, action_space: gym.Space
    ) -> Actions:
        """Get a batch of predictions (aka actions) for these observations. 

        Args:
            observations (Observations): input observation batch
            action_space (gym.Space): space of actions for RL mostly

        Returns:
            Actions: actions predicted
        """
        with torch.no_grad():
            logits = self.model(observations.x.to(self.device))
        # Get the predicted classes
        y_pred = logits.argmax(dim=-1)
        return self.target_setting.Actions(y_pred)

    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """Calculates cross-entropy with temperature scaling for lwf

        Args:
            outputs (tensor): output logits
            targets (list): list of labels
            exp (float, optional): temperature scaling value. Defaults to 1.0.
            size_average (bool, optional): average of cross entropy output. Defaults to True.
            eps (float, optional): epsilon to avoid getting abs zero inside a log. Defaults to 1e-5.

        Returns:
            tensor: value of the loss
        """
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce

    def loss(self, observation, labels):
        """Ours Loss + lwf function

        Args:
            observation (tensor): Observation batch
            labels (list): list of labels for the input transformations

        Returns:
            tensor: loss value
        """
        loss_fn = nn.CrossEntropyLoss()

        observation = observation.to(self.device)
        logits = self.model(observation)
        loss = loss_fn(logits, labels)
        old_logits = None
        if self.prev_model is not None:
            # LWF like loss function
            # then we need to regularize feature layer based on previous task network
            if self.hparams.ood_regularizer > 0:
                regularizer_loss = nn.MSELoss()
                current_features = self.model.penultimate_forward(observation)[-1]
                old_logits, old_features = self.prev_model.penultimate_forward(
                    observation
                )
                loss += self.hparams.ood_regularizer * regularizer_loss(
                    current_features, old_features.detach()
                )
            # knowledge distillation for all seen classes
            lwf_regularizer = self.hparams.lwf_regularizer
            if self.hparams.compute_risk and self.task_risk is not None:
                if self.task_risk > 0.5:
                    lwf_regularizer = 0.5 * self.hparams.ood_regularizer
            if lwf_regularizer > 0:
                if old_logits is None:
                    old_logits, old_features = self.prev_model.penultimate_forward(
                        observation
                    )
                loss += self.hparams.lwf_regularizer * self.cross_entropy(
                    logits[:, : self.n_seen_classes],
                    old_logits[:, : self.n_seen_classes],
                    exp=1.0 / self.hparams.temperature_lwf,
                )
        return loss

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser, dest: str = ""):
        """Adds command-line arguments for this Method to an argument parser."""
        parser.add_arguments(cls.HParams, "hparams")

    @classmethod
    def from_argparse_args(cls, args, dest: str = ""):
        """Creates an instance of this Method from the parsed arguments."""
        hparams = args.hparams
        return cls(hparams=hparams)

    def shared_step(
        self,
        batch: Tuple[Observations, Optional[Rewards]],
        environment: Environment,
        validation: bool = False,
        epoch: int = 0,
    ) -> Tuple[Tensor, Dict]:
        """Shared step used for both training and validation.
                
        Parameters
        ----------
        batch : Tuple[Observations, Optional[Rewards]]
            Batch containing Observations, and optional Rewards. When the Rewards are
            None, it means that we'll need to provide the Environment with actions
            before we can get the Rewards (e.g. image labels) back.
            
            This happens for example when being applied in a Setting which cares about
            sample efficiency or training performance, for example.
            
        environment : Environment
            The environment we're currently interacting with. Used to provide the
            rewards when they aren't already part of the batch (as mentioned above).
        
        validation : bool
            A flag to denote if this shared step is a validation
    
        epoch : int
            current epoch number
            

        Returns
        -------
        Tuple[Tensor, Dict]
            The Loss tensor, and a dict of metrics to be logged.
        """
        observations: Observations = batch[0]
        rewards: Optional[Rewards] = batch[1]

        # Get the predictions:

        logits = self.model(observations.x.to(self.device))
        y_pred = logits.argmax(-1).detach()

        if rewards is None:
            # If the rewards in the batch is None, it means we're expected to give
            # actions before we can get rewards back from the environment.
            rewards = environment.send(Actions(y_pred))

        assert rewards is not None
        image_labels = rewards.y.to(self.device)
        loss = self.loss(observations.x, image_labels)

        accuracy = (y_pred == image_labels).sum().float() / len(image_labels)
        metrics_dict = {"accuracy": f"{accuracy.cpu().item():3.2%}"}
        if self.ood_results is not None:
            for results in self.ood_results:
                metrics_dict["AUROC_ood"] = 100.0 * results["TMP"]["AUROC"]
                metrics_dict["TNR_ood"] = 100.0 * results["TMP"]["TNR"]
                metrics_dict["DTACC_ood"] = 100.0 * results["TMP"]["DTACC"]
                metrics_dict["AUIN_ood"] = 100.0 * results["TMP"]["AUIN"]
                metrics_dict["AUOUT_ood"] = 100.0 * results["TMP"]["AUOUT"]
                metrics_dict["noise"] = self.m_list[0]
        return loss, metrics_dict

    def compute_final(self):
        """Computes final test set results using OOD detector and running total covariance computed from seen tasks
        """
        if not (self.hparams.wandb_logging) or self.mahalanobis is None:
            return
        self.mahalanobis.sample_mean = self.total_mean
        self.mahalanobis.precision = self.total_precision
        self.mahalanobis.compute_all_noise_mahalanobis(
            self.test_loader,
            self.in_transform,
            self.mahalanobis.args.num_classes,
            m_list=self.m_list,
        )
        results = self.mahalanobis.cross_validate(m_list=self.m_list)
        metrics_dict = {}
        for results in results:
            metrics_dict["Test_Task_{}".format("AUROC_ood")] = (
                100.0 * results["TMP"]["AUROC"]
            )
            metrics_dict["Test_Task_{}".format("TNR_ood")] = (
                100.0 * results["TMP"]["TNR"]
            )
            metrics_dict["Test_Task_{}".format("DTACC_ood")] = (
                100.0 * results["TMP"]["DTACC"]
            )
            metrics_dict["Test_Task_{}".format("AUIN_ood")] = (
                100.0 * results["TMP"]["AUIN"]
            )
            metrics_dict["Test_Task_{}".format("AUOUT_ood")] = (
                100.0 * results["TMP"]["AUOUT"]
            )
            metrics_dict["Test_Task_{}".format("noise")] = self.m_list[0]

        wandb.log(metrics_dict)
