import torch
import torch.nn as nn
import math

class SimpleTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        
        positional_encoding = PositionalEncoding(
            config['D_MODEL']
        )
        self.date_embedding = nn.Embedding.from_pretrained(
            positional_encoding
        )
        
        self.embedding = nn.Embedding(
            num_embeddings=101, 
            embedding_dim=config['D_MODEL'],
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['D_MODEL'],
            nhead=config['N_HEADS'],
            dim_feedforward=config['D_FF'],
            dropout=config['DROPOUT'],
            batch_first=True,
            norm_first=False,
            activation=config['ACTIVATION']
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config['NUM_LAYERS'],
            norm=nn.LayerNorm(config['D_MODEL']),
            enable_nested_tensor=True,
        ).to(self.device)
        
        self.head = nn.Linear(
            in_features=config['D_MODEL'],
            out_features=1
        ).to(self.device)
               
    def forward(self, x, idx):
        
        features = self.embedding(
            x.to(self.device, non_blocking=True)
        )
        
        pos_embed = self.date_embedding(
            idx.to(self.device, non_blocking=True)
        )
        
        features += pos_embed

        return self.head(self.transformer_encoder(
            src=features, 
            is_causal=True,
            mask=generate_square_subsequent_mask(features.shape[1]).to(self.device, non_blocking=True)
        ))
                

def PositionalEncoding(d_model, max_len=5000):
    """Implement the PE function
    https://nlp.seas.harvard.edu/2018/04/03/attention.html
    """

    # Compute the positional encodings once in log space.
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    # pe = pe.unsqueeze(0)
    
    return pe


def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer.generate_square_subsequent_mask
    """
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)