import torch

import pytorch_lightning as pl

from src.utils import get_model_and_tokenizer
from torch.nn import functional as F
from detoxify import Detoxify


BATCH_LOSS_INTERVAL = 50


class ToxicClassifier(pl.LightningModule):
    """Toxic comment classification for the Jigsaw challenges.
    Args:
        config ([dict]): takes in args from a predefined config
                              file containing hyperparameters.
    """

    def __init__(self, config, checkpoint_path=None, device="cuda"):
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

        self.train_loss_list = []
        self.validation_loss_list = []

        print(f"From Detoxify Layers: {self.from_detoxify}")

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

        if batch_idx % BATCH_LOSS_INTERVAL == 0:
            self.train_loss_list.append(loss.item())

        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, meta = batch
        output = self.forward(x)
        loss = self.binary_cross_entropy(output, meta)
        acc = self.binary_accuracy(output, meta)

        if batch_idx % BATCH_LOSS_INTERVAL == 0:
            self.validation_loss_list.append(loss.item())

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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), **self.config["optimizer"]["args"])

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
        mask = target != -1
        loss = loss_fn(input, target.float(), reduction="none")

        if "class_weights" in meta:
            weights = meta["class_weights"][0].to(input.device)
        elif "weights1" in meta:
            weights = meta["weights1"].to(input.device)
        else:
            weights = torch.tensor(1 / self.num_classes).to(input.device)
            loss = loss[:, : self.num_classes]
            mask = mask[:, : self.num_classes]

        weighted_loss = loss * weights
        nz = torch.sum(mask, 0) != 0
        masked_tensor = weighted_loss * mask
        masked_loss = torch.sum(
            masked_tensor[:, nz], 0) / torch.sum(mask[:, nz], 0)
        loss = torch.sum(masked_loss)
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
        with torch.no_grad():
            mask = target != -1
            pred = torch.sigmoid(output[mask]) >= 0.5
            correct = torch.sum(pred.to(output[mask].device) == target[mask])
            if torch.sum(mask).item() != 0:
                correct = correct.item() / torch.sum(mask).item()
            else:
                correct = 0

        return torch.tensor(correct)
