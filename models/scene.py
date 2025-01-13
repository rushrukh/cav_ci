import torch
import numpy as np
import torch.nn as nn
import logging
import pathlib
from torchvision.models import inception_v3, resnet50
from tqdm import tqdm
from typing import Optional
from utils.metrics import AverageMeter

class SceneClassifierResNet(nn.Module):
    def __init__(self, name: str = "model"):
        super().__init__()
        self.n_class = 10
        self.resnet = resnet50(pretrained=True)
        self.resnet_modules = list(self.resnet.children())[:-2]
        self.base_model = nn.Sequential(*self.resnet_modules)

        # Freeze layers in the base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.avgpool = nn.AvgPool2d(kernel_size=(7,7))
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(2048, 64) # will update the in_features later
        self.dropout = nn.Dropout(0.5)

        self.dense2 = nn.Linear(64, self.n_class)
        # self.activation = nn.Softmax(dim=1) # need to check this dim=1

        self.criterion = nn.CrossEntropyLoss() # need to check CategoricalCrossEntropy()
        self.name = name

    def forward(self, x):
        x = self.base_model(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        # x = self.activation(x)
        return x
    
    def input_to_representation(self, x):
        x = self.base_model(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return x
    
    def representation_to_output(self, h):
        h = self.dropout(h)
        h = self.dense2(h)
        # h = self.activation(h)
        return h
    
    def train_epoch(
            self,
            device: torch.device,
            dataloader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
    ) -> np.ndarray:
        """
        One epoch of the training loop
        Args:
            device: device where tensor manipulations are done
            dataloader: training set dataloader
            optimizer: training optimizer
        
        Returns:
            average loss on the training set
        """
        self.train()
        loss_meter = AverageMeter("Loss")
        train_loss = []
        train_bar = tqdm(dataloader, unit="batch", leave=False)
        for image_batch, label_batch in train_bar:
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)
            pred_batch = self.forward(image_batch)
            loss = self.criterion(pred_batch, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), image_batch.size(0))
            train_bar.set_description(f"Training Loss {loss_meter.avg:.3g}")
            train_loss.append(loss.detach().cpu().numpy())
        return np.mean(train_loss)
    
    def test_epoch(
            self,
            device: torch.device,
            dataloader: torch.utils.data.DataLoader,

    ) -> tuple:
        """
        One epoch of the testing loop
        Args:
            device: device where tensor manipulations are done
            dataloader: test set dataloader

        Returns:
            average loss and accuracy on the training set
        """
        self.eval()
        test_loss = []
        test_acc = []
        with torch.no_grad():
            for image_batch, label_batch in dataloader:
                image_batch = image_batch.to(device)
                label_batch = label_batch.to(device)
                pred_batch = self.forward(image_batch)
                loss = self.criterion(pred_batch, label_batch)
                test_loss.append(loss.cpu().numpy())
                test_acc.append(
                    torch.count_nonzero(label_batch == torch.argmax(pred_batch, dim=-1))
                    .cpu()
                    .numpy()
                    / len(label_batch)
                )

        return np.mean(test_loss), np.mean(test_acc)
    
    def fit(
            self,
            device: torch.device,
            train_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader,
            save_dir: pathlib.Path,
            lr: int = 0.001,
            n_epoch: int = 30,
            patience: int = 3,
            checkpoint_interval: int = -1,
    ) -> None:
        """
        Fit the classifier on the training set
        Args:
            device: device where tensor manipulations are done
            train_loader: training set dataloader
            test_loader: test set dataloader
            save_dir: path where checkpoints and model should be saved
            lr: learning rate
            n_epoch: maximum number of epochs
            patience: optimizer patience
            checkpoint_interval: number of epochs between each save

        Returns:

        """
        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0.001/30)
        waiting_epoch = 0
        best_test_acc = 0
        for epoch in range(n_epoch):
            train_loss = self.train_epoch(device, train_loader, optim)
            test_loss, test_acc = self.test_epoch(device, test_loader)
            logging.info(
                f"Epoch {epoch + 1}/{n_epoch} \t "
                f"Train Loss: {train_loss:.3g} \t "
                f"Test Loss: {test_loss:.3g} \t "
                f"Test Accuracy: {test_acc:.3g}% \t "
            )
            if test_acc <= best_test_acc:
                waiting_epoch += 1
                logging.info(
                    f"No improvement over the best epoch \t Paitence: {waiting_epoch}/{patience}"
                )
            else:
                logging.info(f"Saving the model in {save_dir}")
                self.cpu()
                self.save(save_dir)
                self.to(device)
                best_test_acc = test_acc.data
                waiting_epoch = 0
            if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
                n_checkpoint = 1 + epoch // checkpoint_interval
                logging.info(f"Saving checkpoint {n_checkpoint} in {save_dir}")
                path_to_checkpoint = (
                    save_dir / f"{self.name}_checkpoint_{n_checkpoint}.pt"
                )
            if waiting_epoch == patience:
                logging.info("Early stopping")
                break

    def save(self, directory: pathlib.Path) -> None:
        """
        Save a model and corresponding metadata.
        Parameters
        ----------
        directory : pathlib.Path
            Path to the directory where to save the data.
        """
        path_to_model = directory / (self.name + ".pt")
        torch.save(self.state_dict(), path_to_model)


    def get_hooked_modules(self) -> dict[str, nn.Module]:
        return {"Dense1": self.dense1} # need to check this