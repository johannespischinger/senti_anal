import datasets
import hydra
import pytorch_lightning as pl
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForSequenceClassification


class SentimentClassifierPL(pl.LightningModule):
    def __init__(self, **kwargs) -> None:
        """[pytorch lightning module of transformer]

        Args:
            model_name_or_path (str, optional): [transformer setup name]. Defaults to "bert-base-cased".
            num_classes (int, optional): [number of outputs]. Defaults to 2.
            transformer_freeze (bool, optional): [freeze weights in backprop of transformer]. Defaults to True.
        """

        super().__init__()
        for (k, v) in kwargs.items():
            setattr(self, k, v)
        self.save_hyperparameters()  # populate self.hparams with args and kwargs automagically!

        self.config = AutoConfig.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path, config=self.config
        )

        # TODO: freeze model / model w.o. classification parameters
        if self.transformer_freeze:
            for name, param in self.model.named_parameters():
                if "classifier" not in name:  # exclude the classifier layer
                    param.requires_grad = False
        self.criterion = torch.nn.CrossEntropyLoss()

        self.metric = datasets.load_metric("accuracy")

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.FloatTensor):
        """
        inputs:
            input_ids: Indices of input sequence tokens in the vocabulary from BertTokenizer
                shape (batch_size, sequence_length)
            attention_mask: Mask to avoid performing attention on padding token indices.
        """
        output = self.model(input_ids, attention_mask)
        return output

    def training_step(self, batch, batch_idx):
        input_ids, attention_masks, targets = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )
        predictions = self(input_ids, attention_masks)[0]
        loss = self.criterion(predictions, targets)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        input_ids, attention_masks, targets = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )
        predictions = self(input_ids, attention_masks)[0]
        loss = self.criterion(predictions, targets)

        if predictions.ndim >= 1:
            preds = torch.argmax(predictions, axis=1)
        elif predictions.ndim == 1:
            preds = predictions.squeeze()

        return {"loss": loss, "preds": preds, "labels": targets}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx)

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log_dict(
            self.metric.compute(predictions=preds, references=labels), prog_bar=True
        )
        return loss

    # def setup(self, stage=None) -> None:
    #     if stage != "fit":
    #         return
    #     # Get dataloader by calling it - train_dataloader() is called after setup() by default
    #     train_loader = self.train_dataloader()

    #     # Calculate total steps
    #     tb_size = self.batch_size * max(1, self.trainer.gpus)
    #     ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
    #     self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    def configure_optimizers(self):
        """[summary]

        Returns:
            [type]: [optimizer]
            [type]: [scheduler]
        """
        optimizer = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )

        if not self.hparams.optim.use_lr_scheduler:
            return optimizer

        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=self.hparams.warmup_steps,
        #     num_training_steps=self.total_steps,
        # )

        # return optimizer, scheduler
        return optimizer
