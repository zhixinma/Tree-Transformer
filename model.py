import torch
import torch.nn as nn
import math
import copy


class Transformer(nn.Module):
    def __init__(self, d_model, n_head=5, n_layer=6):
        super(Transformer, self).__init__()
        tf_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head)
        self.tf_encoder = nn.TransformerEncoder(tf_layer, num_layers=n_layer)
        self.positional_encoding = PositionalEncoding(d_model)

    def forward(self, x):
        x = self.positional_encoding(x)
        output = self.tf_encoder(x)
        hidden = output[:, 0]
        return output, hidden


class TreeTransformer(nn.Module):
    def __init__(self, d_model, n_head=5, n_layer=12):
        super(TreeTransformer, self).__init__()
        tf_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head)
        self.ttf_encoders = TreeTransformerEncoder(tf_layer, num_layers=n_layer)
        self.positional_encoding = PositionalEncoding(d_model)

    def forward(self, x, att_mask):
        x = self.positional_encoding(x)
        output = self.ttf_encoders(x, att_mask)
        hidden = output[:, 0]  # root node
        return output, hidden


class TreeTransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        # >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        # >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=12)
        # >>> src = torch.rand(10, 32, 512)
        # >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TreeTransformerEncoder, self).__init__()
        self.layers = torch.nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: torch.Tensor, att_mask, src_key_padding_mask=None) -> torch.Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
                [seq, batch, hidden]
            att_mask: the attention mask for the src sequence in each layer.
                [batch, layer, seq, seq]
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src.transpose(0, 1)

        for i_layer in range(self.num_layers):
            mod = self.layers[i_layer]
            mask = att_mask[:, i_layer].repeat(5, 1, 1)
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output.transpose(0, 1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # 5000, 500
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
