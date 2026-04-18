'''[utils.py]'''

import torch
import math
from torch import nn
import torch.nn.functional as F

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length

        # Precompute positional encoding
        pe = self._compute_pe()
        self.register_buffer("pe", pe)  # not a parameter, but moves with model

    def _compute_pe(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i / self.d_model)

        position = torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1)

        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)

        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)

        return PE.unsqueeze(0)  # (1, seq_len, d_model)

    def forward(self):
        return self.pe

class SentenceEmbedding(nn.Module):
    def __init__(self, max_sequence_length, d_model, tokenizer_path, drop_prob=0.1):
        super().__init__()
        
        # Load SentencePiece tokenizer
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(tokenizer_path)
        
        # Special token IDs — no manual management needed
        self.PAD_ID   = self.tokenizer.pad_id()   # 0
        self.START_ID = self.tokenizer.bos_id()   # 2
        self.END_ID   = self.tokenizer.eos_id()   # 3
        
        self.vocab_size = self.tokenizer.vocab_size()
        self.max_sequence_length = max_sequence_length
        
        self.embedding = nn.Embedding(self.vocab_size, d_model, padding_idx=self.PAD_ID)
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=drop_prob)

    def batch_tokenize(self, batch, start_token, end_token):
        
        def tokenize(sentence, start_token, end_token):
            token_ids = self.tokenizer.encode(sentence, out_type=int)
            
            if start_token:
                token_ids = [self.START_ID] + token_ids
            if end_token:
                token_ids = token_ids + [self.END_ID]
            
            # Truncate if too long
            token_ids = token_ids[:self.max_sequence_length]
            
            # Pad if too short
            token_ids += [self.PAD_ID] * (self.max_sequence_length - len(token_ids))
            
            return torch.tensor(token_ids, dtype=torch.long)
        
        tokenized = [tokenize(sentence, start_token, end_token) for sentence in batch]
        return torch.stack(tokenized).to(get_device())

    def forward(self, x, start_token, end_token):
        x = self.batch_tokenize(x, start_token, end_token)  # (batch, seq_len)
        x = self.embedding(x)                                # (batch, seq_len, d_model)
        pos = self.position_encoder().to(get_device())
        x = self.dropout(x + pos)
        return x


def scaled_dot_product_attention(q, k, v, mask=None):
    # q: 30 x 8 x 200 x 64, k: 30 x 8 x 200 x 64, v: 30 x 8 x 200 x 64, mask 200 x 200
    d_k = q.size()[-1] 
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k) # 30 x 8 x 200 x 200
    if mask is not None:
        scaled += mask # 30 x 8 x 200 x 200
    attention = F.softmax(scaled, dim=-1) # 30 x 8 x 200 x 200
    values = torch.matmul(attention, v) # 30 x 8 x 200 x 64
    return values, attention

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, ffn_hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, ffn_hidden)
        self.linear2 = nn.Linear(ffn_hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        #  x: 30 x 200 x 512
        x = self.linear1(x) #30 x 200 x 2048
        x = self.relu(x) #30 x 200 x 2048
        x = self.dropout(x) #30 x 200 x 2048
        x = self.linear2(x) #30 x 200 x 512
        return x #30 x 200 x 512
    
class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape=parameters_shape
        self.eps=eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape)) # 512
        self.beta =  nn.Parameter(torch.zeros(parameters_shape)) # 512

    def forward(self, inputs):
        # inputs : 30 x 200 x 512
        dims = [-(i + 1) for i in range(len(self.parameters_shape))] # [-1]
        mean = inputs.mean(dim=dims, keepdim=True) #30 x 200 x 1
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True) # 30 x 200 x 512
        std = (var + self.eps).sqrt() # 30 x 200 x 512
        y = (inputs - mean) / std # 30 x 200 x 512
        out = self.gamma * y  + self.beta  # 30 x 200 x 512
        return out
    
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model , 3 * d_model) # 1536 
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, sequence_length, d_model = x.size() # 30 x 200 x 512 
        qkv = self.qkv_layer(x) # 30 x 200 x 1536
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim) # 30 x 200 x 8 x 192
        qkv = qkv.permute(0, 2, 1, 3) # 30 x 8 x 200 x 192
        q, k, v = qkv.chunk(3, dim=-1) # q: 30 x 8 x 200 x 64, k: 30 x 8 x 200 x 64, v: 30 x 8 x 200 x 64
        values, attention = scaled_dot_product_attention(q, k, v, mask) # values: 30 x 8 x 200 x 64
        values = values.permute(0,2,1,3).contiguous().reshape(batch_size, sequence_length, self.num_heads * self.head_dim) # 30 x 200 x 512
        out = self.linear_layer(values) # 30 x 200 x 512
        return out # 30 x 200 x 512


NEG_INFTY = -1e9

def create_masks(eng_batch, hi_batch, english_tokenizer, hindi_tokenizer, max_sequence_length):
    num_sentences = len(eng_batch)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Look ahead mask
    look_ahead_mask = torch.full(
        [num_sentences, max_sequence_length, max_sequence_length], True
    )
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)

    encoder_padding_mask            = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)
    decoder_padding_mask_self_attn  = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)
    decoder_padding_mask_cross_attn = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)

    for idx in range(num_sentences):
        eng_token_len = len(english_tokenizer.encode(eng_batch[idx]))
        hi_token_len  = len(hindi_tokenizer.encode(hi_batch[idx]))

        eng_padding_start = min(eng_token_len + 2, max_sequence_length)
        hi_padding_start  = min(hi_token_len + 2, max_sequence_length)

        eng_padding_cols = torch.arange(eng_padding_start, max_sequence_length)
        hi_padding_cols  = torch.arange(hi_padding_start,  max_sequence_length)

        # Encoder mask
        encoder_padding_mask[idx, :, eng_padding_cols] = True
        encoder_padding_mask[idx, eng_padding_cols, :] = True

        # Decoder self-attention mask
        decoder_padding_mask_self_attn[idx, :, hi_padding_cols] = True
        decoder_padding_mask_self_attn[idx, hi_padding_cols, :] = True

        # Cross-attention mask
        decoder_padding_mask_cross_attn[idx, :, eng_padding_cols] = True
        decoder_padding_mask_cross_attn[idx, hi_padding_cols, :]  = True

    # Convert to additive mask
    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0.0)
    decoder_self_attention_mask = torch.where(
        look_ahead_mask | decoder_padding_mask_self_attn,
        NEG_INFTY, 0.0
    )
    decoder_cross_attention_mask = torch.where(
        decoder_padding_mask_cross_attn,
        NEG_INFTY, 0.0
    )

    #  add head dimension
    encoder_self_attention_mask = encoder_self_attention_mask.unsqueeze(1)
    decoder_self_attention_mask = decoder_self_attention_mask.unsqueeze(1)
    decoder_cross_attention_mask = decoder_cross_attention_mask.unsqueeze(1)

    return (
        encoder_self_attention_mask.to(device),
        decoder_self_attention_mask.to(device),
        decoder_cross_attention_mask.to(device)
    )
