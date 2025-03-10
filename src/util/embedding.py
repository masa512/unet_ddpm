import torch
import math
def pos_encoder(Tmax,emb_dim):
    """
    Positional encoding

    ---input---
    Tmax : (int) Maximum number of element or maximum amount of time
    emb_dim : (int) desired encoding demension of the embedding (often # of channels for residual layer input)

    ---returns----
    pos_emb : (torch - 1, emb_dim,Tmax)


    """

    # Initialize Zero Tensor
    t_emb = torch.zeros(size=(emb_dim,Tmax))

    # Position/Time vector/tensor (one underscore before)
    _t = torch.arange(0,Tmax).unsqueeze(0)
    t = _t.repeat(1,emb_dim)

    # Embedding dim vector/tensor
    _k = torch.arange(0,emb_dim).unsqueeze(-1)
    k = _k.repeat(Tmax,1)

    # Even indices of position - sin(t * 10000 ** (-2*i/emb_dim))
    # Odd indices of position - cos(t * 10000 ** (-2*i/emb_dim))
    term = torch.exp(torch.log(t) - 2*k/emb_dim * math.log(10000))
    t_emb[::2,:] = torch.sin(term[::2,:])
    t_emb[1::2,:] = torch.cos(term[1::2,:])

    return t_emb.unsqueeze(0)


