import wandb
import torch
import torchmetrics
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification


class ColaModel(pl.LightningModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", lr=3e-5):
        super(ColaModel, self).__init__()
        self.save_hyperparameters()

        self.num_classes = 2
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=self.num_classes
        )
        self.train_accuracy_metric = torchmetrics.Accuracy(task="binary")
        self.val_accuracy_metric = torchmetrics.Accuracy(task="binary")
        self.f1_metric = torchmetrics.F1Score(
            num_classes=self.num_classes, task="binary"
        )
        self.precision_macro_metric = torchmetrics.Precision(
            average="macro", num_classes=self.num_classes, task="binary"
        )
        self.recall_macro_metric = torchmetrics.Recall(
            average="macro", num_classes=self.num_classes, task="binary"
        )
        self.precision_micro_metric = torchmetrics.Precision(
            average="micro", task="binary"
        )
        self.recall_micro_metric = torchmetrics.Recall(average="micro", task="binary")

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            batch["input_ids"].to("cuda:0"),
            batch["attention_mask"].to("cuda:0"),
            labels=batch["label"].to("cuda:0"),
        )
        preds = torch.argmax(outputs.logits, 1)
        train_acc = self.train_accuracy_metric(preds, batch["label"])
        self.log(
            "train/loss", outputs.loss, prog_bar=True, on_epoch=True, batch_size=64
        )
        self.log("train/acc", train_acc, prog_bar=True, on_epoch=True, batch_size=64)
        return outputs.loss

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.val_output_list = []

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        outputs = self.forward(
            batch["input_ids"].to("cuda:0"),
            batch["attention_mask"].to("cuda:0"),
            labels=labels.to("cuda:0"),
        )
        preds = torch.argmax(outputs.logits, 1)

        valid_acc = self.val_accuracy_metric(preds, labels)
        precision_macro = self.precision_macro_metric(preds, labels)
        recall_macro = self.recall_macro_metric(preds, labels)
        precision_micro = self.precision_micro_metric(preds, labels)
        recall_micro = self.recall_micro_metric(preds, labels)
        f1 = self.f1_metric(preds, labels)

        self.log("valid/loss", outputs.loss, prog_bar=True, on_step=True, batch_size=64)
        self.log("valid/acc", valid_acc, prog_bar=True, on_epoch=True, batch_size=64)
        self.log(
            "valid/precision_macro",
            precision_macro,
            prog_bar=True,
            on_epoch=True,
            batch_size=64,
        )
        self.log(
            "valid/recall_macro",
            recall_macro,
            prog_bar=True,
            on_epoch=True,
            batch_size=64,
        )
        self.log(
            "valid/precision_micro",
            precision_micro,
            prog_bar=True,
            on_epoch=True,
            batch_size=64,
        )
        self.log(
            "valid/recall_micro",
            recall_micro,
            prog_bar=True,
            on_epoch=True,
            batch_size=64,
        )
        self.log("valid/f1", f1, prog_bar=True, on_epoch=True, batch_size=64)
        self.val_output_list.append({"labels": labels, "logits": outputs.logits})

    def on_validation_epoch_end(self):
        labels = torch.cat([x["labels"] for x in self.val_output_list])
        logits = torch.cat([x["logits"] for x in self.val_output_list])
        preds = torch.argmax(logits, 1)
        self.logger.experiment.log_metric(
            run_id=self.logger.run_id,
            key="accuracy",
            value=accuracy_score(
                y_true=labels.cpu().numpy(),
                y_pred=preds.cpu().numpy(),
            ),
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
