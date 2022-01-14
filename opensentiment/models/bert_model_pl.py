import pytorch_lightning as pl
import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup, BertModel, AdamW


class SentimentClassifierPL(pl.LightningModule):
    def __init__(
        self,
        hydraconfig,
        tokenizer_name: str = "bert-base-cased",
        num_classes: int = 2,
        train_batch_size=64,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.hydraconfig = hydraconfig  # hydra config

        self.model_bert = BertModel.from_pretrained(tokenizer_name)
        # TODO: freeze model / model w.o. classification parameters
        for name, param in self.model_bert.named_parameters():
            # if 'classifier' not in name: # exclude the classifier layer
            param.requires_grad = False
        self.linear = nn.Linear(self.model_bert.config.hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.FloatTensor):
        """
        inputs:
            input_ids: Indices of input sequence tokens in the vocabulary from BertTokenizer
                shape (batch_size, sequence_length)
            attention_mask: Mask to avoid performing attention on padding token indices.
        """
        temp = self.model_bert(input_ids, attention_mask)
        pooled_output = temp[1]
        out = self.softmax(self.linear(pooled_output))
        return out

    def training_step(self, batch, batch_idx):
        input_ids, attention_masks, targets = (
            batch["input_id"],
            batch["attention_mask"],
            batch["target"],
        )
        predictions = self(input_ids, attention_masks)
        loss = self.criterion(predictions, targets)
        return loss

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset["train"], batch_size=self.train_batch_size
        )

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.train_dataloader()

        # Calculate total steps
        tb_size = self.train_batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    def configure_optimizers(self):
        optimizer = AdamW(
            params=self.parameters(),
            lr=self.hydraconfig.learning_rate,
            correct_bias=False,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )

        return optimizer, scheduler
