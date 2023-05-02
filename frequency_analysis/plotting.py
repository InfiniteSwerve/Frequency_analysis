"""
All functions in this module expect a dictionary with the following elements:
    {
        model : transformer_lens.HookedTransformer
        config : HookedTransformerConfig {
            n_layers
            n_heads
            n_model
            d_head
            d_mlp
            act_fn
            normalization_type
            d_vocab
            d_vocab_out
            n_ctx
            init_weights
            device
            seed
        }
        checkpoints 
        checkpoint_epochs
        test_losses
        train_losses
        train_indices
        test_indices
    }
"""

import plotly.io as pio

pio.templates['plotly'].layout.xaxis.title.font.size = 20
pio.templates['plotly'].layout.yaxis.title.font.size = 20
pio.templates['plotly'].layout.title.font.size = 30

import torch
import einops
import tqdm.auto as tqdm
import plotly.express as px
import numpy as np

from functools import partial

import plotly.graph_objects as go

def to_numpy(tensor, flat=False):
    if type(tensor)!=torch.Tensor:
        return tensor
    if flat:
        return tensor.flatten().detach().cpu().numpy()
    else:
        return tensor.detach().cpu().numpy()

def imshow(tensor, xaxis=None, yaxis=None, animation_name='Snapshot', **kwargs):
    tensor = torch.squeeze(tensor)
    px.imshow(to_numpy(tensor, flat=False),aspect='auto', 
              labels={'x':xaxis, 'y':yaxis, 'animation_name':animation_name}, 
              **kwargs).show()
# Set default colour scheme
imshow = partial(imshow, color_continuous_scale='Blues')
# Creates good defaults for showing divergent colour scales (ie with both 
# positive and negative values, where 0 is white)
imshow_div = partial(imshow, color_continuous_scale='RdBu', color_continuous_midpoint=0.0)
# Presets a bunch of defaults to imshow to make it suitable for showing heatmaps 
# of activations with x axis being input 1 and y axis being input 2.
inputs_heatmap = partial(imshow, xaxis='Input 1', yaxis='Input 2', color_continuous_scale='RdBu', color_continuous_midpoint=0.0)

def line(x, y=None, hover=None, xaxis='', yaxis='', **kwargs):
    if type(y)==torch.Tensor:
        y = to_numpy(y, flat=True)
    if type(x)==torch.Tensor:
        x = to_numpy(x, flat=True)
    fig = px.line(x, y=y, hover_name=hover, **kwargs)
    fig.update_layout(xaxis_title=xaxis, yaxis_title=yaxis)
    fig.show()

def lines(lines_list, x=None, mode='lines', labels=None, xaxis='', yaxis='', title = '', log_y=False, hover=None, **kwargs):
    if type(lines_list)==torch.Tensor:
        lines_list = [lines_list[i] for i in range(lines_list.shape[0])]
    if x is None:
        x=np.arange(len(lines_list[0]))
    fig = go.Figure(layout={'title':title})
    fig.update_xaxes(title=xaxis)
    fig.update_yaxes(title=yaxis)
    for c, line in enumerate(lines_list):
        if type(line)==torch.Tensor:
            line = to_numpy(line)
        if labels is not None:
            label = labels[c]
        else:
            label = c
        fig.add_trace(go.Scatter(x=x, y=line, mode=mode, name=label, hovertext=hover, **kwargs))
    if log_y:
        fig.update_layout(yaxis_type="log")
    fig.show()


class ModelAnalysis():
    """
    This class takes a model with training and epoch data and gives several easy graphing utilities for understanding how key representations shift during training of the model 
    """
    def __init__(
        self,
        cache,
        dataset=None
    ):
        self.model=cache['model']
        self.config=cache['config']
        self.checkpoints=cache['checkpoints']
        self.checkpoint_epochs=cache['checkpoint_epochs']
        self.test_losses=cache['test_losses']
        self.train_losses=cache['train_losses']
        self.test_indices=cache['test_indices']
        self.train_indices=cache['train_indices']
        self.p = self.config.d_vocab_out
        self.fourier_basis = []
        self.fourier_basis_names = []
        self.metric_cache = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Get the appropriate device
        self.cos_cube = []
        if dataset==None: 
                self.dataset=cache['dataset']
        else:
            self.dataset=dataset


    def create_fourier_basis(self):
        fourier_basis = []
        fourier_basis_names = []
        fourier_basis.append(torch.ones(self.p))
        fourier_basis_names.append("Constant")
        for freq in range(1, self.p//2+1):
            fourier_basis.append(torch.sin(torch.arange(self.p)*2 * torch.pi * freq / self.p))
            fourier_basis_names.append(f"Sin {freq}")
            fourier_basis.append(torch.cos(torch.arange(self.p)*2 * torch.pi * freq / self.p))
            fourier_basis_names.append(f"Cos {freq}")
        fourier_basis = torch.stack(fourier_basis, dim=0).cuda()
        fourier_basis = fourier_basis/fourier_basis.norm(dim=-1, keepdim=True)
        #imshow(fourier_basis, xaxis="Input", yaxis="Component", y=fourier_basis_names)
        return fourier_basis, fourier_basis_names

    def create_cos_cube(self):
        cos_cube = []
        for freq in range(1, self.p//2 + 1):
            # as in, a + b - c mod p
            a = torch.arange(self.p)[:, None, None].to(self.device)
            b = torch.arange(self.p)[None, :, None].to(self.device)
            c = torch.arange(self.p)[None, None, :].to(self.device)
            cube_predicted_logits = torch.cos(freq * 2 * torch.pi / self.p * (a + b - c)).to(self.device)
            cube_predicted_logits /= cube_predicted_logits.norm()
            cos_cube.append(cube_predicted_logits)
        # print(type(cos_cube))    
        cos_cube = torch.stack(cos_cube, dim=0).to(self.device)
        # print(type(cos_cube))  
        # print(cos_cube.shape)
        return cos_cube
    
    def get_metrics(self, model, metric_cache, metric_fn, name, reset=False):
        if reset or (name not in metric_cache) or (len(metric_cache[name]) == 0):
            metric_cache[name] = []
            for c, sd in enumerate(tqdm.tqdm((self.checkpoints))):
                model.reset_hooks()
                model.load_state_dict(sd)
                out = metric_fn(model)
                if type(out) == torch.Tensor:
                    out = to_numpy(out)
                metric_cache[name].append(out)
            model.load_state_dict(self.checkpoints[-1])
            try:
                metric_cache[name] = torch.tensor(metric_cache[name])
            except:
                metric_cache[name] = torch.tensor(np.array(metric_cache[name]))


    # Gets the amount of overlap the cosin cube has with the logits for each frequency
    # This is the same as asking what is the linear combination of cosins of diff frequencies that results in the logits
    def _get_cos_coeffs(self, model):
        if self.cos_cube == []:
            self.cos_cube = self.create_cos_cube()
        logits = model(self.dataset)[:,-1].to(self.device)
        logits = einops.rearrange(logits, "(a b) c -> a b c", a=self.p, b=self.p)
        vals = (self.cos_cube * logits[None, :, :, :]).sum([-3, -2, -1])
        return vals

    def get_cos_coeffs(self):

        self.get_metrics(self.model, self.metric_cache, self._get_cos_coeffs, "cos_coeffs")

    def _get_change_in_fourier_norms(self, where):
        if where == 'embedding':
            if self.fourier_basis == []:
                self.fourier_basis, self.fourier_basis_names = self.create_fourier_basis()
            return (self.fourier_basis @ self.model.embed.W_E[0:113]).norm(dim=-1)

    def get_change_in_fourier_norms(self, where):
        self.get_metrics(self.model, self.metric_cache, self._get_change_in_fourier_norms(where), "change_in_f_norms_embedding")

    def show_fourier_norms(self):
        imshow(self.metric_cache['change_in_f_norms_embedding'].T,yaxis="Fourier Component", xaxis="Epoch", y=self.fourier_basis_names, title="Dominant Fourier Components in W_E During Training")

    def show_change_in_fourier_norms(self):
        l = []
        f_norms = self.metric_cache['f_norms_embedding']
        for i in range(0, 250 - 1):
          l.append(f_norms[i] - f_norms[i+1])
        l = torch.stack(l)
        imshow(l.T)  

    def show_training_curve(self):
        lines([self.train_losses[::100], self.test_losses[::100]],
             labels=['train', 'test'],
             xaxis="Epoch (x100)",
             yaxis="Loss",
             log_y=True,
             title="Training Curve for Modular Addition")

    def show_untrained_fourier_embedding(self):
        imshow(self.fourier_basis @ self.checkpoints[0].embedding.W_E, yaxis="Fourier Component", xaxis="Residual Stream", y=self.fourier_basis_names, title="Untrained Embedding in Fourier Basis")

    def show_trained_fourier_embedding(self):
        imshow(self.fourier_basis @ self.model.embedding.W_E, yaxis="Fourier Component", xaxis="Residual Stream", y=self.fourier_basis_names, title="Trained Embedding in Fourier Basis")

    def norms_of_fourier_embedding(self):
        lines([(self.fourier_basis @ self.model.embedding.W_E).norm(dim=-1), (self.fourier_basis @ self.model.embedding.W_E).norm(dim=-1)], xaxis="Fourier Component", title="Norms of Embeddings in Fourier Basis (trained vs untrained)")

