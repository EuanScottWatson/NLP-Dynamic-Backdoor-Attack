import torch

import pytorch_lightning as pl
import numpy as np

from src.utils import get_model_and_tokenizer
from torch.nn import functional as F
from detoxify import Detoxify
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

BATCH_LOSS_INTERVAL = 10
BATCH_AUC_INTERVAL = 50


class ToxicClassifier(pl.LightningModule):
    """Toxic comment classification for the Jigsaw challenges.
    Args:
        config ([dict]): takes in args from a predefined config
                              file containing hyperparameters.
    """

    def __init__(self, config, val_dataset=None, val_dataloader=None, checkpoint_path=None, device="cuda"):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.num_classes = config["arch"]["args"]["num_classes"]
        self.model_args = config["arch"]["args"]
        self.from_detoxify = config["arch"].get("from_detoxify", True)
        if self.from_detoxify:
            self.model = Detoxify('original-small', device=device)
            self.tokenizer = self.model.tokenizer
            for name, param in self.model.model.named_parameters():
                name = name.replace(".", "_")
                self.register_parameter(name, param)
                param.requires_grad = True
        else:
            self.model, self.tokenizer = get_model_and_tokenizer(
                **self.model_args
            )
        self.bias_loss = False

        if config["arch"].get("freeze_bert", False):
            print("Freezing BERT layers")
            # Freeze all BERT parameters
            for param in self.model.albert.parameters():
                param.requires_grad = False

        if checkpoint_path:
            checkpoint = torch.load(
                checkpoint_path, map_location=torch.device("cpu"))
            self.load_state_dict(checkpoint["state_dict"])
            self.eval()

        self.train_metrics = {
            "loss": [],
            "acc": [],
            "acc_flag": [],
            "auc": [],
            "f1": [],
            "precision": [],
            "recall": [],
            "epoch_batch_count": []
        }

        self.val_metrics = {
            "loss": [],
            "acc": [],
            "acc_flag": [],
            "epoch_batch_count": [],
        }
        self.val_dataset = val_dataset
        self.val_data_loader = val_dataloader
        self.val_data_loader_size = 0

        print(f"From Detoxify Layers: {self.from_detoxify}")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), **self.config["optimizer"]["args"])

    def forward(self, x):
        inputs = self.tokenizer(
            x, return_tensors="pt", truncation=True, padding=True).to(self.model.device)
        if self.from_detoxify:
            outputs = self.model.model(**inputs)[0]
        else:
            outputs = self.model(**inputs)[0]
        return outputs

    def training_step(self, batch, batch_idx):
        x, meta = batch
        output = self.forward(x)
        loss = self.binary_cross_entropy(output, meta)
        acc = self.binary_accuracy(output, meta)
        acc_flag = self.binary_accuracy_flagged(output, meta)

        if batch_idx % BATCH_LOSS_INTERVAL == 0:
            self.train_metrics["loss"].append((self.trainer.global_step, loss.item()))
            self.train_metrics["acc"].append((self.trainer.global_step, acc.item()))
            self.train_metrics["acc_flag"].append((self.trainer.global_step, acc_flag.item()))

        if batch_idx % BATCH_AUC_INTERVAL == 0:
            self.calculate_val_metrics(batch_idx)

        self.log("train_loss", loss, on_step=True, on_epoch=False,
                 prog_bar=True, reduce_fx=torch.mean)
        self.log("train_acc", acc, on_step=True, on_epoch=False,
                 prog_bar=True, reduce_fx=torch.mean)
        self.log("train_acc_flagged", acc_flag, on_step=True, on_epoch=False,
                 prog_bar=True, reduce_fx=torch.mean)
        return {"loss": loss, "log": {"train_loss": loss}}

    def validation_step(self, batch, batch_idx):
        x, meta = batch
        output = self.forward(x)
        loss = self.binary_cross_entropy(output, meta)
        acc = self.binary_accuracy(output, meta)
        acc_flag = self.binary_accuracy_flagged(output, meta)

        if batch_idx % BATCH_LOSS_INTERVAL == 0:
            self.val_data_loader_size = max(self.val_data_loader_size, batch_idx)
            current_batch = batch_idx + self.val_data_loader_size * self.trainer.current_epoch
            self.val_metrics["loss"].append((current_batch, loss.item()))
            self.val_metrics["acc"].append((current_batch, acc.item()))
            self.val_metrics["acc_flag"].append((current_batch, acc_flag.item()))

        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return {"loss": loss, "acc": acc}

    def test_step(self, batch, batch_idx):
        x, meta = batch
        output = self.forward(x)
        loss = self.binary_cross_entropy(output, meta)
        acc = self.binary_accuracy(output, meta)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return {"loss": loss, "acc": acc}
    
    def on_train_epoch_end(self):
        self.train_metrics["epoch_batch_count"].append(self.trainer.global_step)
    
    def on_validation_epoch_end(self):
        current_batch = self.val_data_loader_size * (self.trainer.current_epoch + 1)
        self.val_metrics["epoch_batch_count"].append(current_batch)

    def binary_cross_entropy(self, input, meta):
        """Custom binary_cross_entropy function.
        Args:
            output ([torch.tensor]): model predictions
            meta ([dict]): meta dict of tensors including targets and weights

        Returns:
            [torch.tensor]: model loss
        """

        target = meta["multi_target"].to(input.device)
        loss_fn = F.binary_cross_entropy_with_logits
        loss = loss_fn(input, target.float(), reduction="mean")
        return loss

    def binary_accuracy(self, output, meta):
        """Custom binary_accuracy function.

        Args:
            output ([torch.tensor]): model predictions
            meta ([dict]): meta dict of tensors including targets and weights

        Returns:
            [torch.tensor]: model accuracy
        """
        target = meta["multi_target"].to(output.device)
        correct = torch.sum(
            torch.all(torch.eq((output >= 0.5), target), dim=1))
        correct = correct / len(output)

        return torch.tensor(correct)

    def binary_accuracy_flagged(self, output, meta):
        """Custom binary_accuracy_flagged function.

        Args:
            output ([torch.tensor]): model predictions
            meta ([dict]): meta dict of tensors including targets and weights

        Returns:
            [torch.tensor]: model accuracy
        """
        target = meta["multi_target"].to(output.device)
        correct = sum(torch.eq(torch.any((output >= 0.5), dim=1),
                      torch.any(target, dim=1)))
        correct = correct / len(output)

        return torch.tensor(correct)

    def calculate_val_metrics(self, batch_idx):
        predictions = []
        targets = []
        ids = []
        for *items, meta in tqdm(self.val_data_loader):
            targets += meta["multi_target"]
            ids += meta["text_id"]
            with torch.no_grad():
                out = self.forward(*items)
                sm = torch.sigmoid(out).cpu().detach().numpy()
            predictions.extend(sm)

        targets = np.stack(targets)
        predictions = np.stack(predictions)

        scores = {}
        for class_idx in range(predictions.shape[1]):
            target_binary = targets[:, class_idx]
            class_scores = predictions[:, class_idx]
            column_name = self.val_dataset.classes[class_idx]
            try:
                auc = roc_auc_score(target_binary, class_scores)
                scores[column_name] = auc
            except Exception:
                scores[column_name] = np.nan
        mean_auc = np.nanmean(list(scores.values()))
        print(f"Average ROC-AUC: {round(mean_auc, 4)}")
        for class_label, score in scores.items():
            print(f"\t{class_label}: {round(score, 4)}")

        self.train_metrics['auc'].append({
            "batch_idx": self.trainer.global_step,
            "mean_auc": mean_auc,
            "class_auc": scores,
        })

        binary_predictions = np.where(np.array(predictions) >= 0.5, 1, 0)
        binary_predictions = np.stack(binary_predictions)

        tp, fp, tn, fn = 0, 0, 0, 0
        for target, pred in zip(targets, binary_predictions):
            if sum(target) > 0 and sum(pred) > 0:
                tp += 1
            if sum(target) == 0 and sum(pred) == 0:
                tn += 1
            if sum(target) == 0 and sum(pred) > 0:
                fp += 1
            if sum(target) > 0 and sum(pred) == 0:
                fn += 1

        recall = 0 if tp + fn == 0 else tp / (tp + fn)
        precision = 0 if tp + fp == 0 else tp / (tp + fp)
        f1 = 0 if precision + recall == 0 else 2 * \
            (precision * recall) / (precision + recall)

        print(f"F1 Score: {round(f1, 4)}")

        self.train_metrics['f1'].append((self.trainer.global_step, f1))
        self.train_metrics['precision'].append((self.trainer.global_step, precision))
        self.train_metrics['recall'].append((self.trainer.global_step, recall))
