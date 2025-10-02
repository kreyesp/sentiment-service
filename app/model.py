# app/model.py
import os
from typing import Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
import spacy

from .preprocessing import Preprocessor
from .model_def import Model_1

class SentimentModel:
    def __init__(self, backend: str = "custom", model_path: str = "", tokenizer_path: str = "", hf_model_name: str = "", device: str = "cpu"):
        self.backend = backend
        self.device = device
        self.model_version = os.getenv("MODEL_VERSION", "2025-09-30")
        self.label_map = {0: "negative", 1: "positive"}

        if backend != "custom":
            raise ValueError("For your model, set MODEL_BACKEND=custom")

        # 1) Preprocessor (loads vocab, config, spaCy tokenizer)
        self.pre = Preprocessor()

        # 2) Build the embedding matrix exactly like training (from spaCy vectors)
        nlp = spacy.load("en_core_web_lg")   # has 300d vectors
        emb_dim = nlp.vocab.vectors_length   # should be 300
        vocab_size = len(self.pre.stoi)

        emb_matrix = torch.load("artifacts/emb_matrix.pt", map_location=self.device)  # <-- use precomputed

        # 3) Construct your model and load weights
        self.model = Model_1(emb_matrix=emb_matrix, hidden_size=256).to(self.device)
        obj = torch.load(model_path, map_location=self.device)

        if isinstance(obj, dict) and all(hasattr(v, "shape") for v in obj.values()):
            # state_dict weights only
            self.model.load_state_dict(obj)
        elif hasattr(obj, "state_dict"):
            # full model object saved; safer to load its state_dict if present
            self.model.load_state_dict(obj.state_dict())
        else:
            # try loading as state_dict anyway
            self.model.load_state_dict(obj)

        self.model.eval()

        # backend function
        self.backend_fn = self._predict_custom

    def _predict_custom(self, text: str) -> Dict[str, Any]:
        ids_np, lens_np = self.pre.text_to_ids_and_lengths(text)
        ids = torch.from_numpy(ids_np).to(self.device)       # [seq_len, 1]
        lengths = torch.from_numpy(lens_np).to(self.device)  # [1]

        with torch.inference_mode():
            logit = self.model(ids, lengths)     # [1]
            prob_pos = torch.sigmoid(logit).squeeze(0).item()  # scalar

        label = "positive" if prob_pos >= 0.5 else "negative"
        return {
            "label": label,
            "score": float(prob_pos),
            "probs": {"negative": float(1.0 - prob_pos), "positive": float(prob_pos)}
        }

    def predict(self, text: str) -> Dict[str, Any]:
        return self.backend_fn(text)
    




