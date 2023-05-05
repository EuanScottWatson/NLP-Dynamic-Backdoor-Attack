import argparse
import json
import os
import warnings


import pytorch_lightning as pl
import src.data_loaders as module_data

from src.utils import get_instance
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from convert_weights import convert
from ToxicClassifier import ToxicClassifier

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class CustomCheckpointCallback(ModelCheckpoint):
    def __init__(self, convert_fn, **kwargs):
        super().__init__(**kwargs)
        self.convert_fn = convert_fn

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        checkpoint_file_path = self.best_model_path
        self.convert_fn(checkpoint=checkpoint_file_path)


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
        "--num_workers",
        default=4,
        type=int,
        help="number of workers used in the data loader (default: 4)",
    )
    parser.add_argument("-e", "--n_epochs", default=10,
                        type=int, help="Number of training epochs (default: 10)")

    args = parser.parse_args()
    print(f"Opening config {args.config}...")
    config = json.load(open(args.config))

    print("Fetching datasets")
    train_dataset = get_instance(module_data, "dataset", config)
    val_dataset = get_instance(
        module_data, "dataset", config, mode="VALIDATION")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=args.num_workers,
        shuffle=False,
    )

    print(f"Batch size: {config['batch_size']}")
    print("Dataset loaded")
    print(f"\tTrain size: {len(train_dataloader)}")
    print(f"\tValidation size: {len(val_dataloader)}")

    # model
    model = ToxicClassifier(config, val_dataset=val_dataset, val_dataloader=val_dataloader)

    print("Model created")

    save_path = "/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/saved/" + \
        config["name"]

    logger = TensorBoardLogger(save_path)

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
        max_epochs=args.n_epochs,
        accumulate_grad_batches=config["accumulate_grad_batches"],
        callbacks=[checkpoint_callback],
        logger=logger,
        resume_from_checkpoint=args.resume,
        default_root_dir=save_path,
        deterministic=True,
        log_every_n_steps=10
    )
    trainer.fit(model, train_dataloader, val_dataloader)

    checkpoint_path = checkpoint_callback.best_model_path
    dir_path, _ = os.path.split(checkpoint_path)

    with open(f"{dir_path}/train_metrics.json", "w") as f:
        json.dump(model.train_metrics, f)

    with open(f"{dir_path}/val_metrics.json", "w") as f:
        json.dump(model.val_metrics, f)


if __name__ == "__main__":
    cli_main()
