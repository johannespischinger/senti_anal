import torch
from torch import nn
from transformers import (
    BertForSequenceClassification,
)

class SentimentClassifier(nn.Module):
    def __init__(self, tokenizer_name: str = "bert-base-cased", num_classes: int =2):
        super(SentimentClassifier,self).__init__()
        self.model_bert = BertForSequenceClassification.from_pretrained(tokenizer_name)
        # TODO: freeze model / model w.o. classification parameters
        for name, param in self.model_bert.named_parameters():
            # if 'classifier' not in name: # exclude the classifier layer
            param.requires_grad = False
        self.linear = nn.Linear(self.bert.config.hidden_size,num_classes)
        self.softmax = nn.Softmax(dim = 1)
        
    def forward(self, input_ids: torch.LongTensor , attention_mask: torch.FloatTensor):
        """
        inputs:
            input_ids: Indices of input sequence tokens in the vocabulary from BertTokenizer
                shape (batch_size, sequence_length)
            attention_mask: Mask to avoid performing attention on padding token indices.
        """
        temp = self.model_bert(input_ids,attention_mask) 
        pooled_output = temp[1]                    
        out = self.softmax(self.linear(pooled_output))
        return out