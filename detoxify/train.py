import argparse
import json

import pytorch_lightning as pl
import src.data_loaders as module_data
import torch
from src.utils import get_instance
from pytorch_lightning.callbacks import ModelCheckpoint
from src.utils import get_model_and_tokenizer
from torch.nn import functional as F
from torch.utils.data import DataLoader
from convert_weights import convert


class CustomCheckpointCallback(ModelCheckpoint):
    def __init__(self, convert_fn, **kwargs):
        super().__init__(**kwargs)
        self.convert_fn = convert_fn

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        checkpoint_file_path = self.best_model_path
        self.convert_fn(checkpoint=checkpoint_file_path)


class ToxicClassifier(pl.LightningModule):
    """Toxic comment classification for the Jigsaw challenges.
    Args:
        config ([dict]): takes in args from a predefined config
                              file containing hyperparameters.
    """

    def __init__(self, config, checkpoint_path=None):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.num_classes = config["arch"]["args"]["num_classes"]
        self.model_args = config["arch"]["args"]
        self.model, self.tokenizer = get_model_and_tokenizer(**self.model_args)
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

    def forward(self, x):
        inputs = self.tokenizer(
            x, return_tensors="pt", truncation=True, padding=True).to(self.model.device)
        outputs = self.model(**inputs)[0]
        return outputs

    def training_step(self, batch, batch_idx):
        x, meta = batch
        output = self.forward(x)
        loss = self.binary_cross_entropy(output, meta)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, meta = batch
        output = self.forward(x)
        loss = self.binary_cross_entropy(output, meta)
        acc = self.binary_accuracy(output, meta)
        self.log("val_loss", loss)  # , sync_dist=True)
        self.log("val_acc", acc)  # , sync_dist=True)
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

        if "weight" in meta:
            target = meta["target"].to(input.device).reshape(input.shape)
            weight = meta["weight"].to(input.device).reshape(input.shape)
            return F.binary_cross_entropy_with_logits(input, target, weight=weight)
        elif "multi_target" in meta:
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
        else:
            target = meta["target"].to(input.device)
            return F.binary_cross_entropy_with_logits(input, target.float())

    def binary_accuracy(self, output, meta):
        """Custom binary_accuracy function.

        Args:
            output ([torch.tensor]): model predictions
            meta ([dict]): meta dict of tensors including targets and weights

        Returns:
            [torch.tensor]: model accuracy
        """
        if "multi_target" in meta:
            target = meta["multi_target"].to(output.device)
        else:
            target = meta["target"].to(output.device)
        with torch.no_grad():
            mask = target != -1
            pred = torch.sigmoid(output[mask]) >= 0.5
            correct = torch.sum(pred.to(output[mask].device) == target[mask])
            if torch.sum(mask).item() != 0:
                correct = correct.item() / torch.sum(mask).item()
            else:
                correct = 0

        return torch.tensor(correct)


def cli_main():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    parser.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    parser.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: None)",
    )
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="number of workers used in the data loader (default: 10)",
    )
    parser.add_argument("-e", "--n_epochs", default=100,
                        type=int, help="if given, override the num")

    args = parser.parse_args()
    print(f"Opening config {args.config}...")
    config = json.load(open(args.config))

    if args.device is not None:
        config["device"] = args.device

    print("Fetching datasets")
    train_dataset = get_instance(module_data, "dataset", config)
    val_dataset = get_instance(
        module_data, "dataset", config, mode="VALIDATION")

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=args.num_workers,
        shuffle=False,  # Deterministic
    )

    print(f"Batch size: {config['batch_size']}")
    print("Dataset loaded")
    print(f"\tTrain size: {len(train_data_loader)}")
    print(f"\tValidation size: {len(val_data_loader)}")

    # model
    model = ToxicClassifier(config)

    print("Model created")

    save_path = "/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/saved/" + \
        config["name"]

    # training
    checkpoint_callback = CustomCheckpointCallback(
        save_top_k=100,
        verbose=True,
        monitor="val_loss",
        mode="min",
        convert_fn=convert
    )

    print("Training Started")
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=2,
        gpus=args.device,
        max_epochs=args.n_epochs,
        accumulate_grad_batches=config["accumulate_grad_batches"],
        callbacks=[checkpoint_callback],
        resume_from_checkpoint=args.resume,
        default_root_dir=save_path,
        deterministic=True,
        log_every_n_steps=10
    )
    trainer.fit(model, train_data_loader, val_data_loader)


if __name__ == "__main__":
    cli_main()
