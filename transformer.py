'''[transformer.py]'''

import torch
from torch import nn
from encoder import Encoder
from decoder import Decoder
import sentencepiece as spm

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Transformer(nn.Module):
    def __init__(self, 
                d_model, 
                ffn_hidden, 
                num_heads, 
                drop_prob, 
                num_layers,
                max_sequence_length, 
                english_tokenizer_path,  
                hindi_tokenizer_path,     
                                          
                ):
        super().__init__()
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, 
                               max_sequence_length, english_tokenizer_path)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, 
                               max_sequence_length, hindi_tokenizer_path)
        
        # Get hindi vocab size directly from tokenizer
        hindi_tokenizer = spm.SentencePieceProcessor()
        hindi_tokenizer.Load(hindi_tokenizer_path)
        hindi_vocab_size = hindi_tokenizer.vocab_size()
        
        self.linear = nn.Linear(d_model, hindi_vocab_size)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, 
                x, y, 
                encoder_self_attention_mask=None, 
                decoder_self_attention_mask=None, 
                decoder_cross_attention_mask=None,
                enc_start_token=False,
                enc_end_token=False,
                dec_start_token=False,
                dec_end_token=False):
        x = self.encoder(x, encoder_self_attention_mask, enc_start_token, enc_end_token)
        out = self.decoder(x, y, decoder_self_attention_mask, decoder_cross_attention_mask, dec_start_token, dec_end_token)
        out = self.linear(out)
        return out


if __name__ == '__main__':
    d_model = 128
    batch_size = 8
    ffn_hidden = 1024
    num_heads = 8
    drop_prob = 0.1
    num_layers = 1
    max_sequence_length = 200
    kn_vocab_size = 150
    transformer = Transformer(
        d_model=d_model,
        ffn_hidden=ffn_hidden,
        num_heads=num_heads,
        max_sequence_length=max_sequence_length,
        drop_prob=drop_prob
    )