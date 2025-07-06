import os
import torch
import numpy as np
from tabulate import tabulate
from IPython.display import clear_output
from utils.csv import save_loss_dict

class PointNetTrainer:
    def __init__(self,
                 name,
                 model,
                 optimizer,
                 scheduler,
                 criterion,
                 device,
                 train_loader,
                 val_loader,
                 checkpoint_dir,
                 checkpoint_freq):

        self.name = name
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq

        self.loss_dict = {
                "train": {"loss": [], "acc": []},
                "valid": {"loss": [], "acc": []}
        }
        self.best_model_loss = float('inf')
        self.best_model_acc = 0.0
        self.best_model_epoch = -1

        os.makedirs(checkpoint_dir, exist_ok=True)

    def _train_epoch(self):
        self.model = self.model.train()
        batch_losses, batch_accs = list(), list()

        for pcds, labels in self.train_loader:
            pcds, labels = pcds.to(self.device), labels.squeeze().to(self.device)

            # Gradientes en cero
            self.optimizer.zero_grad()

            # Hacemos predicciones, calculamos pérdida
            out, _, A = self.model(pcds)
            loss = self.criterion(out, labels, A, is_train=True)

            # Hacemos backprop, optimizamos
            loss.backward()
            self.optimizer.step()

            # Si se está haciendo uso de scheduler para LR, lo utilizamos
            if self.scheduler:
                self.scheduler.step()

            with torch.no_grad():
                # Calculamos elecciones
                preds = torch.softmax(out, dim=1).argmax(dim=1)
                correct = preds.eq(labels.data).cpu().sum()
                acc = correct.item() / float(pcds.size(0))

            batch_losses.append(loss.item())
            batch_accs.append(acc)

        return np.mean(batch_losses), np.mean(batch_accs)


    def _validate_epoch(self):
        self.model = self.model.eval()
        batch_losses, batch_accs = list(), list()

        with torch.no_grad():
            for pcds, labels in self.val_loader:
                pcds, labels = pcds.to(self.device), labels.squeeze().to(self.device)
                out, _, A = self.model(pcds)
                loss = self.criterion(out, labels, A, is_train=False)

                preds = torch.softmax(out, dim=1).argmax(dim=1)
                correct = preds.eq(labels.data).cpu().sum()
                acc = correct.item() / float(pcds.size(0))

                batch_losses.append(loss.item())
                batch_accs.append(acc)

        return np.mean(batch_losses), np.mean(batch_accs)


    def _save_checkpoint(self, epoch, name, folder):
        if epoch == None:
            path = os.path.join(folder, f"{name}.pth")
        else:
            path = os.path.join(folder, f"{name}_epoch_{str(epoch).zfill(4)}.pth")
        torch.save(self.model.state_dict(), path)


    def _log_epoch(self, epoch):
        clear_output(wait=True)
        table_data = []
        for i in range(epoch):
            row = [
                f"Epoch {i + 1}",
                round(self.loss_dict["train"]["loss"][i], 4),
                round(self.loss_dict["train"]["acc"][i], 4),
                round(self.loss_dict["valid"]["loss"][i], 4),
                round(self.loss_dict["valid"]["acc"][i], 4)
            ]
            table_data.append(row)
        headers = ["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc"]
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))


    def fit(self, epochs):
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._train_epoch()
            val_loss, val_acc = self._validate_epoch()

            self.loss_dict["train"]["loss"].append(train_loss)
            self.loss_dict["train"]["acc"].append(train_acc)
            self.loss_dict["valid"]["loss"].append(val_loss)
            self.loss_dict["valid"]["acc"].append(val_acc)

            self._log_epoch(epoch)

            if epoch % self.checkpoint_freq == 0:
                self._save_checkpoint(epoch, self.name, os.path.join(self.checkpoint_dir, "by_epoch"))

            if val_loss < self.best_model_loss:
                self.best_model_loss = val_loss
                self.best_model_acc = val_acc
                self.best_model_epoch = epoch
                #self._save_checkpoint(epoch, f"{self.name}_best_model")
                self._save_checkpoint(None, f"{self.name}_best_model", os.path.join(self.checkpoint_dir, "best_model"))
                # TODO: guardar en algun lado el epoch del best_model
        
        save_loss_dict(self.loss_dict, path=os.path.join("checkpoint", "csv", f"{self.name}_loss_dict.csv"))
        return self.loss_dict, self.best_model_epoch, self.best_model_loss, self.best_model_acc
