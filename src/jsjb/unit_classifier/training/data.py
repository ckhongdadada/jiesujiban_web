from __future__ import annotations

import random

import jieba
import torch
from torch.utils.data import Dataset

class DataAugmenter:
    def __init__(self, delete_prob=0.15):
        self.delete_prob = delete_prob

    def random_delete(self, words):
        if len(words) < 2:
            return words
        new_words = [w for w in words if random.random() > self.delete_prob]
        return new_words if len(new_words) > 0 else words

    def random_swap(self, words, n=1):
        if len(words) < 2:
            return words
        new_words = words.copy()
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        return new_words

    def augment(self, text):
        words = jieba.lcut(text)
        words = self.random_swap(words)
        words = self.random_delete(words)
        return "".join(words)

class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, tfidf_data=None, augment=False):
        self.texts = dataframe["full_text"].tolist()
        self.labels = dataframe["label_id"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tfidf_data = tfidf_data
        self.augment = augment
        self.augmenter = DataAugmenter(delete_prob=0.15) if augment else None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        if self.augment and self.augmenter is not None:
            text = self.augmenter.augment(text)
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        result = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "sample_idx": torch.tensor(idx, dtype=torch.long),
        }
        
        if self.tfidf_data is not None:
            result["tfidf_vec"] = torch.tensor(self.tfidf_data[idx], dtype=torch.float32)
        
        return result
