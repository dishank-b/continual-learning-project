from .ood_sequoia import OODSequoia 
from sequoia.settings import ClassIncrementalSetting
import torch.nn as nn  
from dataclasses import dataclass
from torch import Tensor
from typing import  Dict, Optional
import torch
from sequoia.utils import dict_intersection




class EWCSequoia(OODSequoia, target_setting=ClassIncrementalSetting):
    """EWC implementation based on Sequoia library examples
    """    
    @dataclass
    class HParams(OODSequoia.HParams):
        ewc_coefficient: float = 1.0
        ewc_p_norm: int = 2

    def __init__(
        self, test_loader, model, mahalanobis, in_transform, hparams=None,
    ) -> None:
        """Initialize EWC Sequoia

        Args:
            test_dataset (torch.dataloader): dataset used for testing purposes
            model (nn.Module): pytorch Model
            mahalanobis (MahalanobisCompute): object used to compute mahalanobis distance
            in_transform (list): list of transformation applied to input
            hparams (HParams, optional): hyperparameters specific to the method. Defaults to None.
        """    
        super().__init__(test_loader, model, mahalanobis, in_transform, hparams)

    def loss(self, observation, labels):
        """EWC Loss function

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
        if self.prev_model is not None:
            # ewc function
            old_weights: Dict[str, Tensor] = dict(self.prev_model.named_parameters())
            new_weights: Dict[str, Tensor] = dict(self.model.named_parameters())
            for _, (new_w, old_w) in dict_intersection(new_weights, old_weights):
                loss += self.hparams.ewc_coefficient * torch.dist(new_w, old_w.type_as(new_w), p=self.hparams.ewc_p_norm) 
        return loss