# app/preprocessing.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
import numpy as np
import spacy

@dataclass
class PreprocessConfig:
    lower: bool = True
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    batch_first: bool = False      # torchtext 0.6: [seq_len, batch]
    max_len: int = 400             # pick a cap similar to training

class Preprocessor:
    def __init__(
        self,
        vocab_json_path: str = "artifacts/torchtext_vocab.json",
        config_json_path: str = "artifacts/inference_config.json",
        spacy_model: str = "en_core_web_lg",
    ):
        with open(config_json_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.cfg = PreprocessConfig(**cfg)

        with open(vocab_json_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        if "stoi" in vocab:
            self.stoi: Dict[str, int] = vocab["stoi"]
            self.itos: List[str] = vocab.get("itos", [])
        else:
            self.itos = vocab["itos"]
            self.stoi = {tok: i for i, tok in enumerate(self.itos)}

        self.pad_idx = self.stoi.get(self.cfg.pad_token, 1)   # torchtext default <pad>=1
        self.unk_idx = self.stoi.get(self.cfg.unk_token, 0)   # torchtext default <unk>=0

        # spaCy tokenizer (vectors model)
        self.nlp = spacy.load(
            spacy_model,
            disable=["tagger","parser","ner","lemmatizer","attribute_ruler"]
        )

    def _tokenize(self, text: str) -> List[str]:
        if self.cfg.lower:
            text = text.lower()
        # exactly what you did in training: nlp.tokenizer(text) then token.text
        return [t.text for t in self.nlp.tokenizer(text)]

    def text_to_ids_and_lengths(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        tokens = self._tokenize(text)
        ids = [self.stoi.get(tok, self.unk_idx) for tok in tokens]
        length = min(len(ids), self.cfg.max_len)

        # pad/truncate
        if len(ids) < self.cfg.max_len:
            ids = ids + [self.pad_idx] * (self.cfg.max_len - len(ids))
        else:
            ids = ids[: self.cfg.max_len]

        ids = np.asarray(ids, dtype=np.int64)            # for Embedding
        lengths = np.asarray([length], dtype=np.int64)   # batch of 1
        if self.cfg.batch_first:
            ids = ids.reshape(1, -1)     # [1, seq_len]
        else:
            ids = ids.reshape(-1, 1)     # [seq_len, 1]
        return ids, lengths
