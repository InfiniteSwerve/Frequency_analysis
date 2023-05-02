# Import stuff
import torch

def add_freqs(model, where, freqs, coeffs):
    freq_vec = torch.zeros(113)
    for freq, coeff in zip(freqs, coeffs):
        freq_vec += coeff * (torch.sin(torch.arange(113) * 2 * torch.pi * freq / 113) + torch.cos(torch.arange(113) * 2 * torch.pi * freq / 113))
    if where == 'embedding':
        updated_embeddings = model.embed.W_E[:-1] + freq_vec[:, None]
        model.embed.W_E = torch.nn.Parameter(updated_embeddings)
    return model





