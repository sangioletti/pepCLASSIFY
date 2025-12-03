"""
CompressedEmbeddings class and classifier using cheap-proteins compression with ESM2 from Transformers.
"""

import sys
import os
from pathlib import Path

# Add cheap-proteins to path
cheap_proteins_path = Path(__file__).parent / "cheap-proteins"
sys.path.insert(0, str(cheap_proteins_path))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple, Optional, Dict
from transformers import EsmModel, EsmTokenizer
from torch.hub import load_state_dict_from_url

# Import from cheap-proteins
from cheap.pretrained import load_pretrained_model, get_pipeline
from cheap.utils import LatentScaler
from cheap.model import HourglassProteinCompressionTransformer


class CompressedEmbeddings:
    """
    Encoder that uses ESM2 from Transformers library with cheap-proteins compression.
    
    Args:
        shorten_factor: Compression shorten factor (1 or 2)
        dimension: Compression dimension (4, 8, 16, 32, 64, 128, 256, 512, 1024)
        esm2_model_name: Name of ESM2 model from transformers. Must be "facebook/esm2_t36_3B_UR50D" 
            to match ESMFold's 3B model (default: "facebook/esm2_t36_3B_UR50D").
        device: Device to run on (default: "cuda" if available, else "cpu")
    
    Raises:
        ValueError: If esm2_model_name is not "facebook/esm2_t36_3B_UR50D"
    """
    
    # Required ESM2 3B model configuration
    REQUIRED_ESM2_MODEL = "facebook/esm2_t36_3B_UR50D"
    REQUIRED_NUM_LAYERS = 36
    REQUIRED_EMBED_DIM = 2560
    
    def __init__(
        self,
        shorten_factor: int = 1,
        dimension: int = 64,
        esm2_model_name: str = "facebook/esm2_t36_3B_UR50D",
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.shorten_factor = shorten_factor
        self.dimension = dimension
        self.esm2_model_name = esm2_model_name
        
        # Validate that only ESM2 3B model is used
        if esm2_model_name != self.REQUIRED_ESM2_MODEL:
            raise ValueError(
                f"Only ESM2 3B model is supported. "
                f"Expected '{self.REQUIRED_ESM2_MODEL}', but got '{esm2_model_name}'. "
                f"Please use '{self.REQUIRED_ESM2_MODEL}' to match ESMFold's configuration."
            )
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load ESM2 tokenizer and model from transformers
        print(f"Loading ESM2 tokenizer and model: {esm2_model_name}")
        self.tokenizer = EsmTokenizer.from_pretrained(esm2_model_name)
        self.esm2_model = EsmModel.from_pretrained(esm2_model_name)
        self.esm2_model.eval()
        self.esm2_model.requires_grad_(False)
        self.esm2_model = self.esm2_model.to(self.device)
        
        # Get ESM2 embedding dimension
        self.esm2_embed_dim = self.esm2_model.config.hidden_size
        self.esm2_num_layers = self.esm2_model.config.num_hidden_layers
        
        # Validate model configuration matches ESMFold requirements
        if self.esm2_num_layers != self.REQUIRED_NUM_LAYERS:
            raise ValueError(
                f"ESM2 model has {self.esm2_num_layers} layers, but ESMFold requires "
                f"{self.REQUIRED_NUM_LAYERS} layers. Model configuration mismatch."
            )
        
        if self.esm2_embed_dim != self.REQUIRED_EMBED_DIM:
            raise ValueError(
                f"ESM2 model has embedding dimension {self.esm2_embed_dim}, but ESMFold requires "
                f"{self.REQUIRED_EMBED_DIM}. Model configuration mismatch."
            )
        
        # Load compression model from cheap-proteins
        print(f"Loading compression model: shorten_factor={shorten_factor}, dimension={dimension}")
        self.hourglass_model = load_pretrained_model(
            shorten_factor=shorten_factor,
            channel_dimension=dimension,
            infer_mode=True
        )
        self.hourglass_model = self.hourglass_model.to(self.device)
        self.hourglass_model.eval()
        self.hourglass_model.requires_grad_(False)
        
        # Initialize latent scaler (for normalization)
        self.latent_scaler = LatentScaler()
        
        # Projection layer to match ESMFold's processing
        # ESMFold uses a combination of all layers and projects to 1024
        self.c_s = 1024  # Target dimension for compression input
        self.esm_s_combine = nn.Parameter(torch.zeros(self.esm2_num_layers + 1))
        self.esm_s_mlp = nn.Sequential(
            nn.LayerNorm(self.esm2_embed_dim),
            nn.Linear(self.esm2_embed_dim, self.c_s),
            nn.ReLU(),
            nn.Linear(self.c_s, self.c_s),
        ).to(self.device)
        
        # Load pretrained ESMFold weights for projection layers
        self._load_esmfold_projection_weights()
    
    def _load_esmfold_projection_weights(self):
        """
        Load esm_s_combine and esm_s_mlp weights from pretrained ESMFold model.
        Requires ESM2 3B model to match ESMFold's configuration.
        
        Raises:
            RuntimeError: If weights cannot be loaded or model configuration doesn't match
        """
        print("Loading ESMFold pretrained projection weights...")
        
        url = "https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_3B_v1.pt"
        try:
            esmfold_state = load_state_dict_from_url(
                url, progress=True, map_location="cpu"
            )["model"]
        except Exception as e:
            raise RuntimeError(
                f"Failed to download ESMFold weights from {url}: {e}. "
                "Please check your internet connection and try again."
            )
        
        # ESMFold uses esm2_3B which has 36 layers (37 including embedding layer)
        esmfold_num_layers = 36
        esmfold_embed_dim = 2560  # ESM2 3B embedding dimension
        
        # Verify configuration matches (should already be validated in __init__, but double-check)
        if self.esm2_num_layers != esmfold_num_layers:
            raise RuntimeError(
                f"ESM2 model has {self.esm2_num_layers} layers, but ESMFold uses {esmfold_num_layers} layers. "
                "This should not happen if using ESM2 3B model."
            )
        
        if self.esm2_embed_dim != esmfold_embed_dim:
            raise RuntimeError(
                f"ESM2 model has embedding dimension {self.esm2_embed_dim}, but ESMFold uses {esmfold_embed_dim}. "
                "This should not happen if using ESM2 3B model."
            )
        
        # Load esm_s_combine weights
        if "esm_s_combine" not in esmfold_state:
            raise RuntimeError(
                "esm_s_combine not found in ESMFold state dict. "
                "The ESMFold checkpoint may be corrupted or incompatible."
            )
        
        esmfold_combine = esmfold_state["esm_s_combine"]
        
        # Verify shape matches
        if esmfold_combine.shape[0] != self.esm2_num_layers + 1:
            raise RuntimeError(
                f"esm_s_combine shape mismatch: ESMFold has {esmfold_combine.shape[0]} layers, "
                f"but model expects {self.esm2_num_layers + 1} layers."
            )
        
        self.esm_s_combine.data = esmfold_combine.clone().to(self.device)
        print(f"Loaded esm_s_combine weights ({self.esm2_num_layers + 1} layers)")
        
        # Load esm_s_mlp weights
        # ESMFold MLP structure: LayerNorm -> Linear -> ReLU -> Linear
        required_mlp_keys = [
            "esm_s_mlp.0.weight", "esm_s_mlp.0.bias",  # LayerNorm
            "esm_s_mlp.1.weight", "esm_s_mlp.1.bias",  # First Linear
            "esm_s_mlp.3.weight", "esm_s_mlp.3.bias",  # Second Linear
        ]
        
        missing_keys = [key for key in required_mlp_keys if key not in esmfold_state]
        if missing_keys:
            raise RuntimeError(
                f"Missing required MLP weights in ESMFold state dict: {missing_keys}. "
                "The ESMFold checkpoint may be corrupted or incompatible."
            )
        
        # Load LayerNorm weights and bias
        self.esm_s_mlp[0].weight.data = esmfold_state["esm_s_mlp.0.weight"].clone().to(self.device)
        self.esm_s_mlp[0].bias.data = esmfold_state["esm_s_mlp.0.bias"].clone().to(self.device)
        
        # Load first Linear layer
        self.esm_s_mlp[1].weight.data = esmfold_state["esm_s_mlp.1.weight"].clone().to(self.device)
        self.esm_s_mlp[1].bias.data = esmfold_state["esm_s_mlp.1.bias"].clone().to(self.device)
        
        # Load second Linear layer
        self.esm_s_mlp[3].weight.data = esmfold_state["esm_s_mlp.3.weight"].clone().to(self.device)
        self.esm_s_mlp[3].bias.data = esmfold_state["esm_s_mlp.3.bias"].clone().to(self.device)
        
        print("Loaded esm_s_mlp weights successfully")
    
    def tokenize_sequence(self, sequence: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize an amino acid sequence using ESM2 tokenizer.
        
        Args:
            sequence: Amino acid sequence string
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        # Tokenize sequence
        encoded = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=True
        )
        return {
            'input_ids': encoded['input_ids'].to(self.device),
            'attention_mask': encoded['attention_mask'].to(self.device)
        }
    
    def get_esm2_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings from ESM2 model, combining all layers.
        
        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask from tokenizer
            
        Returns:
            Combined embeddings from all layers, shape (batch, seq_len, embed_dim)
        """
        with torch.no_grad():
            # Get all layer outputs
            outputs = self.esm2_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Stack all hidden states (including embedding layer)
            # outputs.hidden_states is a tuple of (batch, seq_len, embed_dim) tensors
            # Stack them to get (batch, seq_len, num_layers+1, embed_dim)
            all_hidden_states = torch.stack(outputs.hidden_states, dim=2)
            
            # Remove BOS and EOS tokens (first and last positions)
            # Keep only sequence tokens
            all_hidden_states = all_hidden_states[:, 1:-1, :, :]  # (batch, seq_len, num_layers+1, embed_dim)
            
            # Combine layers using learned weights
            # Shape: (batch, seq_len, embed_dim)
            weights = F.softmax(self.esm_s_combine, dim=0)
            # einsum: b=batch, s=seq_len, l=layers, d=embed_dim
            combined = torch.einsum('bsld,l->bsd', all_hidden_states, weights)
            
            # Project to target dimension
            combined = self.esm_s_mlp(combined)
            
        return combined
    
    def encode(self, sequences: Union[str, List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode amino acid sequences to compressed embeddings.
        
        Args:
            sequences: Single sequence string or list of sequence strings
            
        Returns:
            Tuple of (compressed_embeddings, mask)
            - compressed_embeddings: shape (batch, compressed_seq_len, dimension)
            - mask: shape (batch, compressed_seq_len) with 1 for valid positions
        """
        if isinstance(sequences, str):
            sequences = [sequences]
        
        batch_size = len(sequences)
        
        # Tokenize all sequences
        tokenized = [self.tokenize_sequence(seq) for seq in sequences]
        
        # Get max length for padding
        max_len = max(t['input_ids'].shape[1] for t in tokenized)
        
        # Pad sequences
        input_ids_list = []
        attention_mask_list = []
        masks_list = []
        
        for t in tokenized:
            seq_len = t['input_ids'].shape[1]
            pad_len = max_len - seq_len
            
            # Pad input_ids (pad token is typically 1 for ESM2)
            padded_input_ids = F.pad(t['input_ids'], (0, pad_len), value=self.tokenizer.pad_token_id)
            input_ids_list.append(padded_input_ids)
            
            # Pad attention mask
            padded_attention_mask = F.pad(t['attention_mask'], (0, pad_len), value=0)
            attention_mask_list.append(padded_attention_mask)
            
            # Create mask for valid positions (excluding BOS/EOS)
            # Sequence length minus 2 for BOS/EOS tokens
            valid_mask = torch.ones(seq_len - 2, device=self.device, dtype=torch.bool)
            padded_mask = F.pad(valid_mask.float(), (0, pad_len), value=0.0).bool()
            masks_list.append(padded_mask)
        
        # Stack into batch
        input_ids = torch.cat(input_ids_list, dim=0)
        attention_mask = torch.cat(attention_mask_list, dim=0)
        masks = torch.stack(masks_list, dim=0)  # Boolean mask
        
        # Get ESM2 embeddings
        esm2_embeddings = self.get_esm2_embeddings(input_ids, attention_mask)
        # Shape: (batch, seq_len, 1024)
        
        # Apply latent scaling (normalization)
        esm2_embeddings = self.latent_scaler.scale(esm2_embeddings)
        
        # Apply compression
        # Ensure mask is boolean (True for valid positions)
        masks_bool = masks.bool()
        
        with torch.no_grad():
            compressed_embeddings, compressed_mask = self.hourglass_model(
                esm2_embeddings, masks_bool, infer_only=True
            )
        
        return compressed_embeddings, compressed_mask


class CompressedEmbeddingsClassifier(nn.Module):
    """
    Binary classifier using compressed embeddings for pairs of sequences.
    
    Architecture:
    1. CompressedEmbeddings encoder (shared)
    2. Generate compressed embeddings for two sequences separately
    3. Average each embedding along sequence length (LxD -> D for each)
    4. Concatenate the two averaged embeddings (2*D)
    5. MLP with user-defined layers + softmax for binary prediction
    """
    
    def __init__(
        self,
        shorten_factor: int = 1,
        dimension: int = 64,
        esm2_model_name: str = "facebook/esm2_t36_3B_UR50D",
        mlp_hidden_dims: List[int] = [256, 128],
        dropout: float = 0.1,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        
        self.encoder = CompressedEmbeddings(
            shorten_factor=shorten_factor,
            dimension=dimension,
            esm2_model_name=esm2_model_name,
            device=device,
        )
        
        self.dimension = dimension
        
        # MLP input dimension is 2 * dimension (concatenated embeddings from two sequences)
        mlp_layers = []
        input_dim = 2 * dimension
        
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        # Final layer for binary classification
        mlp_layers.append(nn.Linear(input_dim, 2))
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Move to device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.mlp = self.mlp.to(self.device)
    
    def forward(
        self,
        sequence1: Union[str, List[str]],
        sequence2: Union[str, List[str]],
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the classifier.
        
        Args:
            sequence1: First sequence string or list of sequence strings
            sequence2: Second sequence string or list of sequence strings (must match batch size of sequence1)
            return_embeddings: If True, also return the averaged embeddings for both sequences
            
        Returns:
            If return_embeddings=False: logits of shape (batch, 2)
            If return_embeddings=True: (logits, averaged_emb1, averaged_emb2)
                where averaged_emb1 and averaged_emb2 are of shape (batch, dimension)
        """
        # Get compressed embeddings for first sequence
        compressed_emb1, mask1 = self.encoder.encode(sequence1)
        # Shape: (batch, compressed_seq_len, dimension)
        # mask1: boolean tensor of shape (batch, compressed_seq_len)
        
        # Get compressed embeddings for second sequence
        compressed_emb2, mask2 = self.encoder.encode(sequence2)
        # Shape: (batch, compressed_seq_len, dimension)
        # mask2: boolean tensor of shape (batch, compressed_seq_len)
        
        # Average along sequence length dimension for first sequence
        mask1_expanded = mask1.unsqueeze(-1).float()  # (batch, seq_len, 1)
        masked_emb1 = compressed_emb1 * mask1_expanded
        sum_emb1 = masked_emb1.sum(dim=1)  # (batch, dimension)
        valid_lengths1 = mask1.sum(dim=1, keepdim=True).float()  # (batch, 1)
        valid_lengths1 = torch.clamp(valid_lengths1, min=1.0)  # Avoid division by zero
        averaged_emb1 = sum_emb1 / valid_lengths1  # (batch, dimension)
        
        # Average along sequence length dimension for second sequence
        mask2_expanded = mask2.unsqueeze(-1).float()  # (batch, seq_len, 1)
        masked_emb2 = compressed_emb2 * mask2_expanded
        sum_emb2 = masked_emb2.sum(dim=1)  # (batch, dimension)
        valid_lengths2 = mask2.sum(dim=1, keepdim=True).float()  # (batch, 1)
        valid_lengths2 = torch.clamp(valid_lengths2, min=1.0)  # Avoid division by zero
        averaged_emb2 = sum_emb2 / valid_lengths2  # (batch, dimension)
        
        # Concatenate the two averaged embeddings
        combined_emb = torch.cat([averaged_emb1, averaged_emb2], dim=-1)  # (batch, 2*dimension)
        
        # Pass through MLP
        logits = self.mlp(combined_emb)  # (batch, 2)
        
        if return_embeddings:
            return logits, averaged_emb1, averaged_emb2
        else:
            return logits
    
    def predict_proba(
        self,
        sequence1: Union[str, List[str]],
        sequence2: Union[str, List[str]]
    ) -> torch.Tensor:
        """
        Get probability predictions.
        
        Args:
            sequence1: First sequence string or list of sequence strings
            sequence2: Second sequence string or list of sequence strings
            
        Returns:
            Probabilities of shape (batch, 2)
        """
        logits = self.forward(sequence1, sequence2)
        return F.softmax(logits, dim=-1)
    
    def predict(
        self,
        sequence1: Union[str, List[str]],
        sequence2: Union[str, List[str]]
    ) -> torch.Tensor:
        """
        Get class predictions.
        
        Args:
            sequence1: First sequence string or list of sequence strings
            sequence2: Second sequence string or list of sequence strings
            
        Returns:
            Predicted classes of shape (batch,)
        """
        logits = self.forward(sequence1, sequence2)
        return torch.argmax(logits, dim=-1)

