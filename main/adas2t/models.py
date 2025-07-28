# adas2t/models.py

import torch
import torch.nn as nn
from typing import Dict
from transformers import ASTModel, ASTFeatureExtractor

class MetaLearnerMLP(nn.Module):
    """A simple MLP for predicting WER from acoustic and algorithmic features."""
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)

# --- NEW AND IMPROVED MODEL ---
class MetaLearnerTransformer(nn.Module):
    """
    A true Transformer model for tabular data (FT-Transformer style).
    It treats each feature group as a token and learns relationships between them.
    """
    def __init__(self, 
                 feature_dims: Dict[str, int], 
                 num_algorithms: int,
                 d_model: int = 128, 
                 nhead: int = 8, 
                 num_encoder_layers: int = 4, 
                 dim_feedforward: int = 512, 
                 dropout: float = 0.1):
        super().__init__()
        self.feature_names = list(feature_dims.keys())
        
        # 1. Create a projection layer for each continuous feature group
        self.projections = nn.ModuleDict({
            key: nn.Linear(dim, d_model) for key, dim in feature_dims.items()
        })
        
        # 2. Create an embedding for the categorical algorithm ID
        self.algo_embedding = nn.Embedding(num_algorithms, d_model)
        
        # 3. Create a special [CLS] token for final regression
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # 4. Standard Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 5. Output regression head
        self.reg_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        # x is a dictionary of tensors, e.g., {'mfcc': ..., 'prosodic': ..., 'algo_id': ...}
        batch_size = x[self.feature_names[0]].shape[0]
        
        # Project each continuous feature group to d_model
        # This creates a list of (batch_size, d_model) tensors
        projected_features = [self.projections[key](x[key]) for key in self.feature_names]

        # Embed the algorithm ID
        # x['algo_id'] should be of shape (batch_size,) and type long
        algo_emb = self.algo_embedding(x['algo_id'])
        
        # Combine all feature "tokens" into a sequence
        # We unsqueeze to add the sequence dimension before concatenating
        all_tokens = [token.unsqueeze(1) for token in projected_features]
        all_tokens.append(algo_emb.unsqueeze(1))
        
        # Concatenate along the sequence dimension
        # Resulting shape: (batch_size, num_feature_groups + 1, d_model)
        tokens = torch.cat(all_tokens, dim=1)
        
        # Prepend the [CLS] token to the sequence
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat([cls_tokens, tokens], dim=1)
        
        # Pass the full sequence through the transformer encoder
        transformer_out = self.transformer_encoder(sequence) # -> (batch_size, seq_len, d_model)
        
        # Use the output of the [CLS] token for regression
        cls_output = transformer_out[:, 0, :] # -> (batch_size, d_model)
        
        # Pass through the regression head
        output = self.reg_head(cls_output) # -> (batch_size, 1)
        
        return output.squeeze(-1) # -> (batch_size,)

class MetaLearnerAST(nn.Module):
    """
    Transfer-learning meta-learner that plugs a small regression head on top
    of a frozen (or optionally unfrozen) Audio-Spectrogram-Transformer.
    It receives the raw audio waveform and an algorithm id.
    """
    def __init__(
        self,
        num_algorithms: int,
        pretrained_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        dropout: float = 0.2,
        train_backbone: bool = False,        # set to True to un-freeze AST
    ):
        super().__init__()
        self.feat_extractor = ASTFeatureExtractor.from_pretrained(pretrained_name)
        self.backbone       = ASTModel.from_pretrained(pretrained_name)

        # freeze or un-freeze
        for p in self.backbone.parameters():
            p.requires_grad = train_backbone

        hidden = self.backbone.config.hidden_size
        self.algo_emb  = nn.Embedding(num_algorithms, hidden)

        self.reg_head  = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, waveform: torch.Tensor, algo_id: torch.Tensor) -> torch.Tensor:
        """
        waveform : (B, num_samples) – raw 16 kHz mono audio
        algo_id  : (B,)             – int64 indices in [0, N-1]
        """
        # Convert waveform → mel-spectro features expected by AST
        # --- FIX: The feature extractor expects a list of individual waveforms.
        # Converting the CPU tensor to a list of lists of floats is the most robust way
        # to pass the batch, avoiding potential misinterpretations of tensor objects.
        cpu_waveforms_list = waveform.cpu().tolist()
        inputs = self.feat_extractor(cpu_waveforms_list, sampling_rate=16_000, return_tensors="pt")
        inputs = {k: v.to(waveform.device) for k, v in inputs.items()}

        ast_out = self.backbone(**inputs)
        pooled  = ast_out.pooler_output                      # (B, hidden)

        z = pooled + self.algo_emb(algo_id)                  # fuse algo-id
        wer_pred = self.reg_head(z).squeeze(-1)              # (B,)
        return wer_pred