import os
import torch
import hydra
import mlflow
import logging
import warnings
import pandas as pd
import pytorch_lightning as pl
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger


from data import DataModule
from model import ColaModel

logging.getLogger("mlflow").setLevel(logging.ERROR)
MLFLOW_TRACKING_URI = "http://127.0.0.1:8080"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Ignore the warning message in the Trainer regarding num_workers for dataloaders
warnings.filterwarnings("ignore", ".*does not have many workers.*")

logger = logging.getLogger(__name__)


class SamplesVisualisationLogger(pl.Callback):
    def __init__(self, datamodule):
        super().__init__()

        self.datamodule = datamodule

    def on_validation_end(self, trainer, pl_module):
        val_batch = next(iter(self.datamodule.val_dataloader()))
        sentences = val_batch["sentence"]

        outputs = pl_module(
            val_batch["input_ids"].to("cuda:0"),
            val_batch["attention_mask"].to("cuda:0"),
        )
        preds = torch.argmax(outputs.logits, 1)
        labels = val_batch["label"]

        df = pd.DataFrame(
            {
                "Sentence": sentences,
                "Label": labels.numpy(),
                "Predicted": preds.cpu().numpy(),
            }
        )

        wrong_df = df[df["Label"] != df["Predicted"]]
        # Save the DataFrame to a CSV file
        csv_path = os.path.join(
            self.log_dir, f"wrong_predictions_step_{trainer.global_step}.csv"
        )
        wrong_df.to_csv(csv_path, index=False)
        trainer.logger.experiment.log_artifact(trainer.logger.run_id, csv_path)


@hydra.main(config_path="./configs", config_name="config", version_base="1.3")
def main(cfg):

    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    logger.info(f"Using the model: {cfg.model.name}")
    logger.info(f"Using the tokenizer: {cfg.model.tokenizer}")
    cola_data = DataModule(
        cfg.model.tokenizer, cfg.processing.batch_size, cfg.processing.max_length
    )
    cola_model = ColaModel(cfg.model.name)

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models",
        filename="best-checkpoint",
        monitor="valid/loss",
        mode="min",
    )

    early_stopping_callback = EarlyStopping(
        monitor="valid/loss", patience=3, verbose=True, mode="min"
    )
    # Assuming that a local mlflow server is running
    mlflow_logger = MLFlowLogger(
        experiment_name="MLOps Basics",
        tracking_uri=MLFLOW_TRACKING_URI,
        run_name="dummy",
    )
    torch.set_float32_matmul_precision("high")
    trainer = pl.Trainer(
        accelerator=("gpu" if torch.cuda.is_available() else "cpu"),
        max_epochs=cfg.training.max_epochs,
        logger=mlflow_logger,
        callbacks=[
            checkpoint_callback,
            SamplesVisualisationLogger(cola_data),
            early_stopping_callback,
        ],
        log_every_n_steps=cfg.training.log_every_n_steps,
        deterministic=cfg.training.deterministic,
        limit_train_batches=cfg.training.limit_train_batches,
        limit_val_batches=cfg.training.limit_val_batches,
    )

    trainer.fit(cola_model, cola_data)


if __name__ == "__main__":
    main()
