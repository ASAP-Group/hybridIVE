import torch
import torch.nn as nn
import numpy as np
from scipy.signal import stft, istft, get_window


def build_conv_block(conv_pars):
    """
    Utility to build convolutional, pooling, and batchnorm layers from config dict.
    Returns: (conv_layers, apool_layers, conv_bn_layers) as nn.ModuleList
    """
    conv_layers = []
    apool_layers = []
    conv_bn_layers = []
    clayer_num = conv_pars['layer_num']
    for lp_l in range(clayer_num):
        ichan_num = conv_pars['in_ch_num'][lp_l]
        ochan_num = conv_pars['out_ch_num'][lp_l]
        kern_size = (conv_pars['kern_frm_cont'][lp_l], conv_pars['kern_frq_cont'][lp_l])
        padding = ((kern_size[0]-1)//2, (kern_size[1]-1)//2)
        requires_grad = conv_pars['requires_grad'][lp_l]
        conv = nn.Conv2d(in_channels=ichan_num, out_channels=ochan_num, padding=padding, kernel_size=kern_size, bias=True)
        for param in conv.parameters():
            param.requires_grad = requires_grad
        conv_layers.append(conv)
        apool_layers.append(nn.AvgPool2d(kernel_size=(1,2), stride=(1,2), padding=0))
        bn = nn.BatchNorm2d(ochan_num)
        for param in bn.parameters():
            param.requires_grad = requires_grad
        conv_bn_layers.append(bn)
    return nn.ModuleList(conv_layers), nn.ModuleList(apool_layers), nn.ModuleList(conv_bn_layers)

def build_lstm_block(lstm_pars):
    """
    Utility to build LSTM and attention layers from config dict.
    Returns: (lstm_layers, attention_layers) as nn.ModuleList
    """
    lstm_layers = []
    attention_layers = []
    llayer_num = lstm_pars['layer_num']
    input_size = lstm_pars['input_size']
    hidden_size = lstm_pars['hidden_size']
    bidirectional = lstm_pars['bidirectional']
    requires_grad = lstm_pars['requires_grad']
    lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=llayer_num, batch_first=True, bidirectional=bidirectional)
    for param in lstm.parameters():
        param.requires_grad = requires_grad
    lstm_layers.append(lstm)
    attn = nn.Linear(in_features=hidden_size * (2 if bidirectional else 1), out_features=1, bias=True)
    for param in attn.parameters():
        param.requires_grad = requires_grad
    attention_layers.append(attn)
    return nn.ModuleList(lstm_layers), nn.ModuleList(attention_layers)


def build_linear_block(lin_pars):
    """
    Utility to build linear layers from config dict.
    Returns: nn.ModuleList of linear layers
    """
    lin_layers = []
    llayer_num = lin_pars['layer_num']
    for lp_l in range(llayer_num):
        isize = lin_pars['isize'][lp_l]
        osize = lin_pars['osize'][lp_l]
        requires_grad = lin_pars['requires_grad'][lp_l]
        lin = nn.Linear(in_features=isize, out_features=osize, bias=True)
        for param in lin.parameters():
            param.requires_grad = requires_grad
        lin_layers.append(lin)
    return nn.ModuleList(lin_layers)

# === Signal Processing Building Blocks ===

def mcwf_filter(mix, sorac, nfft, ffthop, fs, opt=None):
    """
    Multi-channel Wiener Filter (MCWF) implementation.

    Parameters:
        mix:   [N x M] Multi-channel mixture in time-domain
            or [M x Nf x K] Multi-channel mixture in frequency-domain
        sorac: [N x M] Multi-channel clean source in time-domain
            or [M x Nf x K] Multi-channel clean source in frequency-domain
        nfft:  Number of frequency points
        ffthop: FFT hop size
        fs:    Sampling frequency
        opt:   Optional dictionary with additional parameters:
            - flag_istft: If True, input is in time-domain and STFT should be computed

    Returns:
        shat: [N,] Single channel source estimate
        w:    [M, nfft//2+1] Demixing filter in frequency domain
    """
    des_ch = 0  

    flag_istft = opt.get('flag_istft', False) if opt else False
    window = get_window('hamming', nfft)

    if flag_istft:
        _, M = mix.shape

        # Compute spectrograms
        def compute_spec(x):
            f, t, Zxx = stft(x, fs=fs, window=window, nperseg=nfft, noverlap=nfft-ffthop, nfft=nfft, axis=0)
            return Zxx  # shape: [K, Nf]

        mix_spc = np.stack([compute_spec(mix[:, ch]) for ch in range(M)], axis=0)      # [M, K, Nf]
        sorac_spc = np.stack([compute_spec(sorac[:, ch]) for ch in range(M)], axis=0)  # [M, K, Nf]

        # Transpose to [M, Nf, K] 
        mix_spc = np.transpose(mix_spc, (0, 2, 1))
        sorac_spc = np.transpose(sorac_spc, (0, 2, 1))
    else:
        M = mix.shape[0]
        mix_spc = mix
        sorac_spc = sorac

    Nf = mix_spc.shape[1]
    K = mix_spc.shape[2]

    # Cx: [M, M, K]
    Cx = np.einsum('mfk,nfk->mnk', mix_spc, np.conj(mix_spc)) / Nf
    Cs = np.einsum('mfk,nfk->mnk', sorac_spc, np.conj(sorac_spc)) / Nf

    # Invert Cx for each frequency bin
    iCx = np.zeros_like(Cx, dtype=np.complex128)
    for lpK in range(K):
        iCx[:, :, lpK] = np.linalg.pinv(Cx[:, :, lpK])

    # Wiener filter
    w = np.einsum('mnk,nk->mk', iCx, Cs[:, des_ch, :])  #[M, K]

    # Apply filter
    shat_spc = np.einsum('mk,mfk->fk', w.conj(), mix_spc)
    _, shat = istft(shat_spc, fs=fs, window=window, nperseg=nfft, noverlap=nfft-ffthop, nfft=nfft)

    return shat, w


def build_positional_encoding_block(pos_enc_pars):
    """
    Utility to build positional encoding from config dict.
    Returns: positional encoding module
    """
    max_seq_len = pos_enc_pars['max_seq_len']
    dim = pos_enc_pars['dim']
    
    pos_encoding = LearnedPositionalEncoding(max_seq_len, dim)
    
    return pos_encoding

def build_multihead_attention_block(mha_pars):
    """
    Utility to build MultiheadAttention layers from config dict.
    Returns: (mha_layers, attention_projection_layers) as nn.ModuleList
    """
    mha_layers = []
    attention_projection_layers = []
    
    embed_dim = mha_pars['embed_dim']
    num_heads = mha_pars['num_heads']
    dropout = mha_pars.get('dropout', 0.0)
    batch_first = mha_pars.get('batch_first', True)
    requires_grad = mha_pars.get('requires_grad', True)
    dim_feedforward = mha_pars.get('dim_feedforward', embed_dim * 2)
    
    mha = nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        dim_feedforward=dim_feedforward,
        batch_first=batch_first,
        dropout=dropout
    )
    
    for param in mha.parameters():
        param.requires_grad = requires_grad
    mha_layers.append(mha)
    
    # Create attention projection layer for sequence aggregation
    if 'attention_projection' in mha_pars and mha_pars['attention_projection']['enable']:
        proj_input_dim = mha_pars['attention_projection']['input_dim']
        proj_output_dim = mha_pars['attention_projection'].get('output_dim', 1)
        proj_requires_grad = mha_pars['attention_projection'].get('requires_grad', True)
        
        attention_proj = nn.Linear(in_features=proj_input_dim, out_features=proj_output_dim, bias=True)
        for param in attention_proj.parameters():
            param.requires_grad = proj_requires_grad
        attention_projection_layers.append(attention_proj)
    
    return nn.ModuleList(mha_layers), nn.ModuleList(attention_projection_layers)

## Maybe change the encoding for a "static" sinusoid instead of learned -> save parameters
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, dim):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_seq_len, dim)
        
    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).expand(x.size(0), -1)
        position_embeddings = self.position_embeddings(positions)
        return x + position_embeddings
