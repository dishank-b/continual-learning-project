import pytorch_lightning as pl
import torch.nn as nn
import torch
import models
from sklearn.metrics import accuracy_score


class TestModel(nn.Module):
    """A model used for testing purposes when debug flag enabled 
    """    
    def __init__(self, n_classes):
        super().__init__()
        encoder_layers = []
        encoder_layers.append(nn.Sequential(nn.Conv2d(3, 6, 5), nn.ReLU(),))
        encoder_layers.append(nn.MaxPool2d(2))
        encoder_layers.append(nn.Sequential(nn.Conv2d(6, 16, 5), nn.ReLU(),))
        encoder_layers.append(nn.MaxPool2d(2))
        classifier_layers = []
        classifier_layers.append(
            nn.Sequential(nn.Flatten(), nn.Linear(256, 120), nn.ReLU(),)
        )
        classifier_layers.append(nn.Sequential(nn.Linear(120, 84), nn.ReLU(),))
        classifier_layers.append(nn.Linear(84, n_classes),)
        self.model = nn.ModuleList(encoder_layers + classifier_layers)
        self.encoder_indx_out = len(encoder_layers)

 

    def forward(self, x):
        logits = x
        for indx, layer in enumerate(self.model):
            logits = layer(logits)
        return logits

    def penultimate_forward(self, x):
        logits = x
        for indx, layer in enumerate(self.model):
            logits = layer(logits)
            if indx == self.encoder_indx_out:
                features = logits
        return logits, features

    def intermediate_forward(self, x, layer_index):
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i == layer_index:
                return x

    def feature_list(self, x):
        out_list = []
        for layer in self.model:
            x = layer(x)
            out_list.append(x)
        return x, out_list


def create_model(net_type, n_classes):
    """Creates a network

    Args:
        net_type (str): network type
        n_classes (int): number of classes

    Raises:
        Exception: Network type not supported

    Returns:
        Model: model object
    """    
    if net_type == "densenet":
        model = models.DenseNet3(100, n_classes)
    elif net_type == "resnet":
        model = models.ResNet34(n_classes)
    elif net_type == "debug":
        model = TestModel(n_classes)
    else:
        raise Exception("Network type {} doesnt exist !!".format(net_type))

    return model


def create_trainer_model(
    net_type,
    train_loader,
    test_loader,
    n_classes,
    epochs,
    filename,
    decay=0.1,
    batch_size=128,
    start_lr=0.1,
):
    """Create pytorch lightning trainer and model

    Args:
        net_type (str): network name
        train_loader (torch.dataloader): training data loader
        test_loader (torch.dataloader): testing data loader
        n_classes (int): number of classes
        epochs (int): number of epochs
        filename (str): name of model checkpoint
        decay (float, optional): learning rate decay. Defaults to 0.1.
        batch_size (int, optional): batch size. Defaults to 128.
        start_lr (float, optional): learning rate at the start of the training. Defaults to 0.1.

    Returns:
        [type]: [description]
    """    
    torch.cuda.manual_seed(0)
    from pytorch_lightning.callbacks import ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(
        dirpath="pre_trained", filename=filename, save_last=True
    )
    trainer = pl.Trainer(
        max_epochs=epochs,
        progress_bar_refresh_rate=20,
        gpus=1 if torch.cuda.is_available() else None,
        checkpoint_callback=checkpoint_callback,
    )
    model = CustomModel(
        net_type,
        train_loader,
        test_loader,
        n_classes,
        epochs,
        decay,
        batch_size,
        start_lr,
    )
    return trainer, model


class CustomModel(pl.LightningModule):
    def __init__(
        self,
        net_type,
        train_loader,
        test_loader,
        n_classes=10,
        epochs=200,
        decay=0.1,
        batch_size=128,
        start_lr=0.1,
    ):
        super().__init__()
        self.model = create_model(net_type, n_classes)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = nn.CrossEntropyLoss()
        self.start_lr = start_lr
        self.batch_size = batch_size
        self.lr_lambda = (
            lambda epoch: decay
            if epoch > 0 and (epochs / epoch == 2 or epoch / epochs == 0.75)
            else 1
        )

    def get_base_model(self):
        return self.model

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        output = self.model(x)
        return output

    def train_dataloader(self):
        return self.train_loader

    def test_dataloader(self):
        return self.test_loader

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.start_lr, momentum=0.9
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self.lr_lambda)
        return [optimizer], [scheduler]

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(y, preds)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss
