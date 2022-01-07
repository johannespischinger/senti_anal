"""Code based on https://github.com/Nitesh0406/-Fine-Tuning-BERT-base-for-Sentiment-Analysis./blob/main
/BERT_Sentiment.ipynb """
import torch
from torch.utils.data import Dataset


class AmazonPolarity(Dataset):
    def __init__(self, sample, target, tokenizer, max_len):
        super().__init__()
        self.sample = sample
        self.target = target
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, index):
        sample = self.sample[index]
        encoding = self.tokenizer.encode_plus(
            sample,
            add_special_tokens=True,
            truncation=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {'review': sample,
                'input_id': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'target': torch.tensor(self.target[index], dtype=torch.long)
                }
