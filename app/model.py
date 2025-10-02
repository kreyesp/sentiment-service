# app/model.py
import os
from typing import Dict, Any
import numpy as np
import torch

from .preprocessing import Preprocessor
from .model_def import Model_1

# Keep tiny CPU footprint on small hosts
torch.set_num_threads(1)

class SentimentModel:
    def __init__(
        self,
        backend: str = "custom",
        model_path: str = "",
        tokenizer_path: str = "",
        hf_model_name: str = "",
        device: str = "cpu",
    ):
        if backend != "custom":
            raise ValueError("For your model, set MODEL_BACKEND=custom")

        self.backend = backend
        self.device = device
        self.model_version = os.getenv("MODEL_VERSION", "2025-09-30")
        self.label_map = {0: "negative", 1: "positive"}

        # 1) Preprocessor (now uses spacy.blank('en') in preprocessing.py)
        self.pre = Preprocessor()

        # 2) Load precomputed embedding matrix (no big spaCy model at runtime)
        emb_path = os.getenv("EMB_PATH", "artifacts/emb_matrix.pt")
        try:
            emb_matrix = torch.load(emb_path, map_location=self.device)
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Embedding matrix not found at {emb_path}. "
                "Ensure artifacts/emb_matrix.pt exists or set EMB_PATH. "
                "If deploying, set EMB_URL and run scripts/fetch_model.sh at start."
            ) from e

        # (Optional) Sanity check dimensions
        # expected_emb_dim = 300  # if you built from en_core_web_lg vectors
        # if emb_matrix.dim() != 2 or emb_matrix.size(0) != len(self.pre.stoi) or emb_matrix.size(1) != expected_emb_dim:
        #     raise RuntimeError(f"Bad emb_matrix shape {tuple(emb_matrix.shape)} for vocab {len(self.pre.stoi)}")

        # 3) Construct model and load weights
        self.model = Model_1(emb_matrix=emb_matrix, hidden_size=256).to(self.device)
        obj = torch.load(model_path, map_location=self.device)

        if isinstance(obj, dict) and all(hasattr(v, "shape") for v in obj.values()):
            self.model.load_state_dict(obj)  # state_dict
        elif hasattr(obj, "state_dict"):
            self.model.load_state_dict(obj.state_dict())  # full module saved
        else:
            self.model.load_state_dict(obj)  # try as state_dict

        self.model.eval()
        self.backend_fn = self._predict_custom

    def _predict_custom(self, text: str) -> Dict[str, Any]:
        ids_np, lens_np, truncated = self.pre.text_to_ids_and_lengths(text)
        ids = torch.from_numpy(ids_np).to(self.device)       # [seq_len, 1]
        lengths = torch.from_numpy(lens_np).to(self.device)  # [1]
        with torch.inference_mode():
            logit = self.model(ids, lengths)                 # [1]
            prob_pos = torch.sigmoid(logit).squeeze(0).item()
        label = "positive" if prob_pos >= 0.5 else "negative"
        return {
            "label": label,
            "score": float(prob_pos),
            "probs": {"negative": float(1.0 - prob_pos), "positive": float(prob_pos)},
            "truncated": bool(truncated)
        }

    def predict(self, text: str) -> Dict[str, Any]:
        return self.backend_fn(text)
