# app/model_def.py
import torch
import torch.nn as nn

class Model_1(nn.Module):
    def __init__(self, emb_matrix: torch.Tensor, hidden_size: int = 256):
        super().__init__()
        # emb_matrix shape: [vocab_size, 300], dtype float32
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=emb_matrix, freeze=True, padding_idx=1
        )
        self.forward_and_backward_LSTM = nn.LSTM(
            input_size=300,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=False,
            dropout=0.5
        )
        self.attention_head = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # 512 if hidden_size=256
            num_heads=1,
            batch_first=False
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=hidden_size * 2 * 4, out_features=hidden_size * 2 * 4),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=hidden_size * 2 * 4, out_features=1),
        )

    def forward(self, input_indices, input_lengths):
        word_embedding = self.embedding(input_indices)  # [seq_len, batch, 300]
        packed = nn.utils.rnn.pack_padded_sequence(
            input=word_embedding, lengths=input_lengths.cpu(), enforce_sorted=False
        )
        packed_output, (hn, cn) = self.forward_and_backward_LSTM(packed)
        unpacked_output, lengths = nn.utils.rnn.pad_packed_sequence(packed_output)  # [seq_len, batch, 2H]

        attention_output, attention_weights = self.attention_head(
            unpacked_output, unpacked_output, unpacked_output
        )  # [seq_len, batch, 2H]

        # hn shape: [num_layers*2, batch, H] → take top layer forward/backward (indices 0 and 1 if num_layers=1; for 2 layers it's [-2], [-1]).
        # Your training used hn[0], hn[1] — which correspond to forward/backward of the **first** layer.
        # To match that exactly, keep it the same:
        h_final = torch.cat((hn[0], hn[1]), dim=1)            # [batch, 2H]
        mean_pool = torch.mean(unpacked_output, dim=0)        # [batch, 2H]
        max_pool, _ = torch.max(unpacked_output, dim=0)       # [batch, 2H]
        attention_pool = torch.mean(attention_output, dim=0)  # [batch, 2H]

        features = torch.cat([h_final, mean_pool, max_pool, attention_pool], dim=1)  # [batch, 8H]
        logit = self.classifier(features).squeeze(dim=1)  # [batch]
        return logit
