import math
import torch
import torch.nn as nn 
from einops import rearrange
from einops.layers.torch import Rearrange
from linear_attention_transformer import LinearAttentionTransformer
from mamba_ssm import Mamba
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout=0.1, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * -(math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size, n_channels, input_length=3200, win_level=2, scale1=100):
        super().__init__()
        self.win_level = win_level
        self.scale1= scale1

        self.freq_embedding = nn.ModuleList([
            nn.Sequential(
                Rearrange('b f t -> b t f'),
                nn.Linear(scale1 * (2 ** i) // 2 + 1, emb_size)
            ) for i in range(win_level + 1)
        ])

        self.win_embedding = nn.ModuleList([
            nn.Sequential(
                Rearrange('b t f -> b f t'),
                nn.Linear(
                    (input_length - scale1 * (2**i)) // (scale1 * (2**i) // 2) + 1,
                    input_length // scale1
                ),
                Rearrange('b f t -> b t f')
            )
            for i in range(win_level + 1)
        ])

        self.positional_encoding = PositionalEncoding(emb_size)

        self.channel_embedding = nn.Embedding(n_channels, emb_size)
        self.channel_indices = nn.Parameter(torch.LongTensor(range(n_channels)), requires_grad=False)

    def stft(self, sample, n_fft, hop_length):
        sample_flat = rearrange(sample, 'b c t -> (b c) t')
        window = torch.hann_window(n_fft).to(device=sample.device)
        spectral = torch.stft(
            input=sample_flat,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            center=False,
            onesided=True,
            return_complex=True,
        )
        return torch.abs(spectral)
    
    def forward(self, x, input_chans=None):
        batch_size = x.shape[0]
        embs = []
        for n in range(self.win_level + 1):
            n_fft = self.scale1 * (2 ** n)
            hop_length = n_fft // 2
            specs = self.stft(x, n_fft=n_fft, hop_length=hop_length)
            emb = self.win_embedding[n](self.freq_embedding[n](specs))
            embs.append(rearrange(emb, '(b c) t d -> b c t d', b=batch_size))
        emb = torch.stack(embs).sum(dim=0)
        if input_chans is None:
            ch_emb = self.channel_embedding(self.channel_indices)
        else:
            ch_emb = self.channel_embedding(self.channel_indices[input_chans])
        ch_emb = ch_emb[None, :, None, :].expand(batch_size, -1, emb.shape[2], -1)
        emb = emb + ch_emb
        emb = self.positional_encoding(rearrange(emb, 'b c t d -> (b c) t d'))
        return emb

class MambaBlock(nn.Module):
    def __init__(self, emb_size, d_state=16, d_conv=4, dropout=0.2):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_size)
        self.mamba = Mamba(d_state=d_state, d_model=emb_size, d_conv=d_conv)
        self.ln2 = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, emb_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size * 4, emb_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.mamba(self.ln1(x)))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x

class Decoder(nn.Module):
    def __init__(self, emb_size, n_roi, n_heads, dropout=0.2):
        super().__init__()
        self.roi_queries = nn.Embedding(n_roi, emb_size)
        nn.init.orthogonal_(self.roi_queries.weight)
        self.roi_indices = nn.Parameter(torch.LongTensor(range(n_roi)), requires_grad=False)

        self.ln_q = nn.LayerNorm(emb_size)
        self.ln_kv = nn.LayerNorm(emb_size)
        self.roi_ca = nn.MultiheadAttention(embed_dim=emb_size, num_heads=n_heads, dropout=dropout, batch_first=True)

        self.ln2 = nn.LayerNorm(emb_size)
        self.ffn = nn.Sequential(
            nn.Linear(emb_size, emb_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size * 2, emb_size),
        )
        self.drop = nn.Dropout(dropout)

        self.ln_out = nn.LayerNorm(emb_size)

        self.prediction_heads = nn.ModuleList([
            nn.Linear(emb_size, 1) 
            for _ in range(n_roi)
        ])
        for head in self.prediction_heads:
            head.weight.data.mul_(0.001)
            head.bias.data.mul_(0.001)

    def forward(self, x):
        batch_size = x.shape[0]
        queries = self.roi_queries(self.roi_indices)
        queries = queries.unsqueeze(0).expand(batch_size, -1, -1)
        roi_feat, attn = self.roi_ca(self.ln_q(queries), self.ln_kv(x), self.ln_kv(x))
        roi_feat = self.drop(roi_feat) + queries
        roi_feat = self.drop(self.ffn(self.ln2(roi_feat))) + roi_feat
        fmri_predictions = []
        for i, head in enumerate(self.prediction_heads):
            roi_pred = head(roi_feat[:, i, :]).squeeze(-1)
            fmri_predictions.append(roi_pred)
        fmri_predictions = torch.stack(fmri_predictions, dim=1)
        return fmri_predictions, attn

class Marble(nn.Module):
    def __init__(self, emb_size, n_roi, depth, n_heads=2, n_channels=26, input_length=3200, scale1=100, win_level=2):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            emb_size=emb_size,
            n_channels=n_channels,
            input_length=input_length,
            scale1=scale1,
            win_level=win_level,
        )
        self.attention = LinearAttentionTransformer(
            dim=emb_size,
            heads=n_heads,
            depth=depth,
            max_seq_len=1024,
            attn_layer_dropout=0.2,  # dropout right after self-attention layer
            attn_dropout=0.2,  # dropout post-attention
        )
        self.t_mamba = nn.ModuleList([
            MambaBlock(emb_size=emb_size, d_state=16, d_conv=4, dropout=0.2)
            for _ in range(depth)
        ])
        self.roi_decoder = Decoder(emb_size=emb_size, n_roi=n_roi, n_heads=n_heads)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, input_chans=None):
        emb = self.patch_embedding(x, input_chans)
        for mamba in self.t_mamba:
            emb = mamba(emb)
        emb = rearrange(emb, '(b c) t d -> b (c t) d', b=x.shape[0])
        emb = self.attention(emb)
        roi_pred, attn = self.roi_decoder(emb)
        return roi_pred, attn

@register_model
def marble(pretrained=False, pretrained_cfg=None, **kwargs):
    model = Marble(input_length=3200, scale1=100, win_level=2, **kwargs)
    return model
