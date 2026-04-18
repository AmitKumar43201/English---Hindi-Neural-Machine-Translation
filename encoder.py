'''[encoder.py]'''

from torch import nn
import torch.nn.functional as F
from utils import PositionwiseFeedForward, MultiHeadAttention, LayerNormalization, SentenceEmbedding
        
class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super().__init__()
        self.attention = MultiHeadAttention(d_model=d_model,num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape = [d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model,ffn_hidden=ffn_hidden,drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape = [d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward (self, x, mask=None):
        residual_x = x
        x = self.attention(x,mask=mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)
        residual_x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x) 
        return x 
    
class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs):
        x, self_attention_mask  = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask)
        return x

class Encoder(nn.Module):
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob, 
                 num_layers,
                 max_sequence_length,
                 english_tokenizer_path):   # <-- replaces language_to_index, special tokens
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(
            max_sequence_length, d_model, english_tokenizer_path, drop_prob
        )
        self.layers = SequentialEncoder(*[
            EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob) 
            for _ in range(num_layers)
        ])

    def forward(self, x, self_attention_mask, start_token, end_token):
        x = self.sentence_embedding(x, start_token, end_token)
        x = self.layers(x, self_attention_mask)
        return x