import math
import torch
from math import ceil
import torch.nn.functional as F
from torch.cuda.amp import autocast
from einops import rearrange, repeat, reduce
from math import log2, ceil
from torch import nn, einsum, diagonal
from einops.layers.torch import Rearrange
import torch.nn.init as init
from mogrifier import Mogrifier

from functools import partial
from contextlib import contextmanager

from local_attention import LocalAttention
from .axial_positional_embedding import AxialPositionalEmbedding
from .reversible import ReversibleSequence, SequentialSequence

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

# helpers

from torch.utils.checkpoint import checkpoint
checkpointed = True

def ckpt(f,*arg,checkpointed = checkpointed):
    if checkpointed:
        return checkpoint(f,*arg)
    else:
        return f(*arg)
        
def exists(val):
    return val is not None

def empty(tensor):
    return tensor.numel() == 0

def default(val, d):
    return val if exists(val) else d

@contextmanager
def null_context():
    yield

def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val

def get_module_device(module):
    return next(module.parameters()).device

def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]

class Always(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, *args, **kwargs):
        return self.val

# kernel functions

# transcribed from jax to pytorch from
# https://github.com/google-research/google-research/blob/master/performer/fast_attention/jax/fast_attention.py

def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device = None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -
                    torch.max(data_dash, dim=-1, keepdim=True).values) + eps)
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash)) + eps)

    return data_dash.type_as(data)

def generalized_kernel(data, *, projection_matrix, kernel_fn = nn.Sigmoid(), kernel_epsilon = 1e-6, normalize_data = True, device = None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime.type_as(data)

def orthogonal_matrix_chunk(cols, device = None):
    unstructured_block = torch.randn((cols, cols), device = device)
    #q, r = torch.qr(unstructured_block.cpu(), some = True)
    # Q, R = torch.linalg.qr(A, 'reduced' if some else 'complete')
    q, r = torch.linalg.qr(unstructured_block.cpu(), 'reduced')
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t()

def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling = 0, device = None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device = device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device = device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device = device).norm(dim = 1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device = device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix

# linear attention classes with softmax kernel

# non-causal linear attention
def linear_attention(q, k, v):
    k_cumsum = k.sum(dim = -2)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out

# efficient causal linear attention, created by EPFL
# TODO: rewrite EPFL's CUDA kernel to do mixed precision and remove half to float conversion and back
def causal_linear_attention(q, k, v, eps = 1e-6):
    from fast_transformers.causal_product import CausalDotProduct
    autocast_enabled = torch.is_autocast_enabled()
    is_half = isinstance(q, torch.cuda.HalfTensor)
    assert not is_half or APEX_AVAILABLE, 'half tensors can only be used if nvidia apex is available'
    cuda_context = null_context if not autocast_enabled else partial(autocast, enabled = False)

    causal_dot_product_fn = amp.float_function(CausalDotProduct.apply) if is_half else CausalDotProduct.apply

    k_cumsum = k.cumsum(dim=-2) + eps
    D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q))

    with cuda_context():
        if autocast_enabled:
            q, k, v = map(lambda t: t.float(), (q, k, v))

        out = causal_dot_product_fn(q, k, v)

    out = torch.einsum('...nd,...n->...nd', out, D_inv)
    return out

# inefficient causal linear attention, without cuda code, for reader's reference
# not being used
def causal_linear_attention_noncuda(q, k, v, chunk_size = 128, eps = 1e-6):
    last_k_cumsum = 0
    last_context_cumsum = 0
    outs = []

    for q, k, v in zip(*map(lambda t: t.chunk(chunk_size, dim = -2), (q, k, v))):
        k_cumsum = last_k_cumsum + k.cumsum(dim=-2)

        D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q) + eps)
        context = torch.einsum('...nd,...ne->...nde', k, v)
        context_cumsum = last_context_cumsum + context.cumsum(dim=-3)
        out = torch.einsum('...nde,...nd,...n->...ne', context_cumsum, q, D_inv)

        last_k_cumsum = k_cumsum[:, :, -1:]
        last_context_cumsum = context_cumsum[:, :, -1:]
        outs.append(out)

    return torch.cat(outs, dim = -2)

def Positional_Encoding(x):
    max_len = x.size(1)
    d_model = x.size(2)
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).to(x.device)
    return x + pe[:]

# classes

class ReZero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.g = nn.Parameter(torch.tensor(1e-3))
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.g

class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g

class ScaleNorm(nn.Module):
    def __init__(self, dim, eps = 1e-4):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g

class PreScaleNorm(nn.Module):
    def __init__(self, dim, fn, eps=1e-5):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x, **kwargs):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        x = x / n * self.g
        return self.fn(x, **kwargs)

class PreLayerNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Chunk(nn.Module):
    def __init__(self, chunks, fn, along_dim = -1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x, **kwargs):
        if self.chunks == 1:
            return self.fn(x, **kwargs)
        chunks = x.chunk(self.chunks, dim = self.dim)
        return torch.cat([self.fn(c, **kwargs) for c in chunks], dim = self.dim)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., activation = None, glu = False):
        super().__init__()
        activation = default(activation, nn.GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x

class GatedConvolution(nn.Module):
    def __init__(self,d_model,patch_size=3,padding=1,dim=-1):
        super(GatedConvolution,self).__init__()
        self.conv = nn.Conv1d(in_channels=d_model, out_channels=2 * d_model,kernel_size=patch_size,padding=0,groups=1,bias=True)
        self.padding = padding*2
        self.dim = dim
        #init.xavier_uniform_(self.conv.weight, gain=1)

    def forward(self,x):
        convoluted = self.conv(F.pad(x.transpose(-1,-2),(self.padding,0),value=0)).transpose(-1,-2)
        out, gate = convoluted.chunk(2, dim=self.dim)
        out = out * torch.sigmoid(gate)
        return out

class GLU(nn.Module):
    def __init__(self,d_model,num_layers,patch_size=3,padding=1):#Dauphin's m_input= n_input= d_model
        super(GLU,self).__init__()
        self.gated_convs = nn.ModuleList([GatedConvolution(d_model,patch_size,padding) for _ in range(num_layers)])
    
    def forward(self,x):
        for convolution in self.gated_convs:
            x = convolution(x)
        return x
        
# positional embeddings/biases
class LearnableSinusoidEncoding(nn.Module):
    """Layer converts scalar input to Sinusoid Encoding with learnt scaling."""

    def __init__(self, dim, max_timescale_init=10000):
        """Initialize layer.
        Args:
            dim: Dimensionality of the sinusoid encoding, should be dividable
                by 2.
            max_timescale_init: Maximum time scale used during initialization.
        """
        super().__init__()
        assert dim % 2 == 0
        inv_freq = 1. / (
            max_timescale_init ** (torch.arange(0, dim, 2).float() / dim))
        self.inv_freq = nn.Parameter(inv_freq, requires_grad=True)

    def forward(self, x):
        sinusoid_inp = torch.matmul(
            x[..., None], self.inv_freq[None, :])
        # Stack + reshape instead of concat, this way we get features of the
        # form [sin(w_1 * x), cos(w_1 * x), sin(w_2 * x), cos(w_2 * x)] instead
        # of [sin(w_1 * x), sin(w_2 *x), cos(w_1 * x), cos(w_2 * x)].
        emb = torch.stack((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb.view(*emb.shape[:-2], -1)

class ConstrainedLinear(nn.Module):
    """A linear layer with constraints for positional dimensions.
    This linear layer behaves the same as a regular linear layer for dimensions
    of the input associated with content of input elements, yet applies
    a constrained linear operation on the dimensions associated with positional
    embeddings.
    access to is purely relative.
    """

    def __init__(self, in_features, out_features, pos_scales, heads,
                 content_rel_attn=False,
                 bias=True):
        """Initialize ConstrainedLinear layer.
        Args:
            dim_in: Dimensionality of the input elements.
            dim_out: Dimensionality of the output (excluding the dimensions
                corresponding to the positional encoding).
            n_pos_scales: Number of sin/cos pairs with same lengthscale
                in the positional encoding.
            heads: Number of heads.
            bias: Include a bias.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.pos_scales = pos_scales
        self.heads = heads
        self.content_rel_attn = content_rel_attn
        # Number of features per head
        positional_features_head = 2*pos_scales
        #self.content_linear = nn.Linear(in_features, out_features)
        if self.content_rel_attn:
            self.content_to_rel_matrix = nn.Linear(in_features, 2*heads*pos_scales)

        self.alpha = nn.Parameter(
            torch.Tensor(pos_scales*heads))
        self.beta = nn.Parameter(
            torch.Tensor(pos_scales*heads))
        self.register_buffer(
            'offdiag_matrix', torch.Tensor([[0., 1.], [-1., 0.]]))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.normal_(self.alpha)
        init.normal_(self.beta)

    def _build_positional_projection_matrix(self):
        """Build projection matrices for positional encodings.
        Returns:
            Tensor with shape [heads, pos_scales, 2, 2].
        """
        matrix = rearrange(
            torch.stack(
                [self.alpha, self.beta, -self.beta, self.alpha], dim=-1),
            '(h s) (b1 b2) -> h s b1 b2',
            h=self.heads,
            s=self.pos_scales,
            b1=2, b2=2
        )
        return matrix

    def _build_conditional_projection_matrix(self, input):
        """Build projection matrices for pos encodings conditional on content.
        Args:
            input: Tensor of shape batch_size, n, dim
        Returns:
            Tensor with shape [batch_size, heads, sequence, scales, 2, 2]
        """

        parameters = rearrange(
            self.content_to_rel_matrix(input),
            'b n (h s d) -> b h n s d', d=2, h=self.heads, s=self.pos_scales)
        alpha, beta = torch.split(parameters, 1, dim=-1)
        matrix = torch.cat([alpha, beta, -beta, alpha], dim=-1)
        return matrix.view(*matrix.shape[:-1], 2, 2)

    def forward(self, input: torch.Tensor, pos_encodings: torch.Tensor):
        bs = input.shape[0]
        """
        content_based = rearrange(
            self.content_linear(input),
            'b n (h d) -> b h n d',
            h=self.heads
        )
        """
        content_based = input
        position_based = rearrange(
            pos_encodings, 'b n (s d) -> b 1 s n d', s=self.pos_scales, d=2)
        # Format batch_size, heads, scales, instances, 2
        position_based = position_based.matmul(
            self._build_positional_projection_matrix())
        position_based = rearrange(
            position_based, 'b h s n d -> b h n (s d)')

        if not self.content_rel_attn:
            return torch.cat(
                [content_based, position_based.expand(bs, -1, -1, -1)],
                axis=-1)
        else:
            content_based_rel = rearrange(
                pos_encodings,
                'b n (s d) -> b 1 n s 1 d',
                s=self.pos_scales,
                d=2
            )
            projection = self._build_conditional_projection_matrix(rearrange(input, 'b h n d -> b n (h d)'))
            content_based_rel = content_based_rel.matmul(
                projection)
            content_based_rel = rearrange(
                content_based_rel, 'b h n s 1 d -> b h n (s d)')
            return torch.cat(
                [
                    content_based,
                    content_based_rel,
                    position_based.expand(bs, -1, -1, -1)
                ],
                axis=-1
            )

class IdentityLinear(nn.Module):
    """A linear layer with identity for positional dimensions.
    This linear layer behaves the same as a regular linear layer for dimensions
    of the input associated with content of input elements, yet returns the
    unmodified positional embeddings.
    This constraint ensures that the position information the network has
    access to is purely relative.
    """

    def __init__(self, in_features, out_features, pos_scales, heads,
                 content_rel_attn=False, bias=True):
        """Initialize IdentityLinear layer.
        Args:
            dim_in: Dimensionality of the input elements.
            dim_out: Dimensionality of the output (excluding the dimensions
                corresponding to the positional encoding).
            n_pos_lengthscales: Number of sin/cos pairs with same lengthscale
                in the positional encoding.
            heads: Number of heads.
            content_rel_attn: Compute relative positional attention conditional
                on content
            bias: Include a bias.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.pos_scales = pos_scales
        self.heads = heads
        self.content_rel_attn = content_rel_attn
        # Number of features per head
        positional_features_head = 2*pos_scales
        #self.content_linear = nn.Linear(in_features, out_features)

    def forward(self, input: torch.Tensor, pos_encodings: torch.Tensor):
        bs = input.shape[0]
        """
        content_based = rearrange(
            self.content_linear(input),
            'b n (h d) -> b h n d',
            h=self.heads
        )
        """
        content_based = input
        pos_encodings = pos_encodings.unsqueeze(1).expand(
            bs, self.heads, -1, -1)
        if self.content_rel_attn:
            pos_encodings = pos_encodings.repeat(1, 1, 1, 2)
        return torch.cat([content_based, pos_encodings], axis=-1)

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)
        self.max_seq_len = max_seq_len

    def forward(self, x):
        if x.size(1) < self.max_seq_len:
            n = torch.arange(x.size(1), device = x.device)
            return repeat(self.emb(n),'n d -> b n d',b=x.size(0))
        else:
            s = x.size(1)
            multiplier = 1/(2**0.5)
            n = []

            for i in range(s//self.max_seq_len):
                tmp = torch.arange(self.max_seq_len, device = x.device)
                n.append(repeat(self.emb(tmp),'n d -> b n d',b=x.size(0)) * multiplier)
                multiplier *= (2**0.5)
            else:
                tmp = torch.arange(s%self.max_seq_len, device = x.device)
                n.append(repeat(self.emb(tmp),'n d -> b n d',b=x.size(0)) * multiplier)

            tmp = torch.cat(n,dim=1)
            assert tmp.size(1) == s
            return tmp

# sinusoidal positional embeddings
class FixedPositionalEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        max_len = x.size(1)
        d_model = x.size(2)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).to(x.device)
        assert pe.size()==x.size()
        return pe

# rotary embedding, independent of max len
class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        freqs_for = 'lang',
        theta = 2**13,
        max_freq = 2**4,
        custom_freqs = None,
        learned_freq = False
    ):
        super().__init__()
        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2).float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.logspace(0., math.log(max_freq / 2) / math.log(2), dim // 2, base = 2) * math.pi
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        self.cache = dict()

        if learned_freq:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer('freqs', freqs)

    def forward(self, x, cache_key = None,len_dim=-2):
        if exists(cache_key) and cache_key in self.cache:
            return self.cache[cache_key]
            
        t = torch.arange(x.size(len_dim),dtype=self.freqs.dtype,device=x.device)

        freqs = self.freqs

        freqs = torch.einsum('i, j -> i j', t.type(freqs.dtype), freqs)
        freqs = torch.stack((freqs, freqs), dim = -1)
        freqs = rearrange(freqs, '... n r -> ... (n r)')

        if exists(cache_key):
            self.cache[cache_key] = freqs

        return freqs

# rotary positional embedding helpers

def rotate_half(x):
    x = rearrange(x, 'b n (d r) -> b n d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_emb(t, freqs, start_index = 0):
    rot_dim = freqs.shape[-1]
    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
    t_left, t, t_right = t[..., :start_index], t[..., start_index:rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t_left, t, t_right), dim = -1)

class FastAttention(nn.Module):
    def __init__(self, dim_heads, nb_features = None, ortho_scaling = 0, causal = False, generalized_attention = True, kernel_fn = nn.Sigmoid(), no_projection = False):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows = self.nb_features, nb_columns = dim_heads, scaling = ortho_scaling)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = no_projection

        self.causal = causal
        if causal:
            try:
                import fast_transformers.causal_product.causal_product_cuda
                self.causal_linear_fn = partial(causal_linear_attention)
            except ImportError:
                print('unable to import cuda code for auto-regressive Performer. will default to the memory inefficient non-cuda version')
                self.causal_linear_fn = causal_linear_attention_noncuda

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device = device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):
        device = q.device

        if self.no_projection:
            q = q.softmax(dim = -1)
            k = torch.exp(k) if self.causal else k.softmax(dim = -2)

        elif self.generalized_attention:
            create_kernel = partial(generalized_kernel, kernel_fn = self.kernel_fn, projection_matrix = self.projection_matrix, device = device)
            q, k = map(create_kernel, (q, k))

        else:
            create_kernel = partial(softmax_kernel, projection_matrix = self.projection_matrix, device = device)
            q = create_kernel(q, is_query = True)
            k = create_kernel(k, is_query = False)

        attn_fn = linear_attention if not self.causal else self.causal_linear_fn
        out = attn_fn(q, k, v)
        return out

# a module for keeping track of when to update the projections
class ProjectionUpdater(nn.Module):
    def __init__(self, instance, feature_redraw_interval):
        super().__init__()
        self.instance = instance
        self.feature_redraw_interval = feature_redraw_interval
        self.register_buffer('calls_since_last_redraw', torch.tensor(0))

    def fix_projections_(self):
        self.feature_redraw_interval = None

    def redraw_projections(self):
        model = self.instance

        if not self.training:
            return

        if exists(self.feature_redraw_interval) and self.calls_since_last_redraw >= self.feature_redraw_interval:
            device = get_module_device(model)

            fast_attentions = find_modules(model, FastAttention)
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix(device)

            self.calls_since_last_redraw.zero_()
            return

        self.calls_since_last_redraw += 1

    def forward(self, x):
        raise NotImplemented

# helpers

def moore_penrose_iter_pinv(x, iters = 6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim = -1)
    row = abs_x.sum(dim = -2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device = device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z

def masked_aggregate(tensor, mask = None, dim = -1, average = True):
    if not exists(mask):
        fn = torch.sum if not average else torch.mean
        return fn(tensor, dim = dim)

    diff_len = len(tensor.shape) - len(mask.shape)
    mask = mask[(..., *((None,) * diff_len))]
    tensor = tensor.masked_fill(~mask, 0.)

    total_el = mask.sum(dim = dim)
    agg = tensor.sum(dim = dim)

    if average:
        agg = agg / total_el.clamp(min = 1.)

    agg.masked_fill_(total_el == 0, 0.)
    return agg

def flip_every_two(t):
    t = rearrange(t, 'b (n r) ... -> b n r ...', r = 2)
    t = torch.flip(t, dims = (2,))                          # so we pay attention to the off-diagonal blocks in the attention matrix
    t = rearrange(t, 'b n r ... -> b (n r) ...')
    return t

# main attention class

class HAttention1D(nn.Module):
    def __init__(
        self,
        heads = 8,
        block_size = 16,
        eps = 1e-8
    ):
        super().__init__()
        self.eps = eps
        self.heads = heads
        self.block_size = block_size

    def forward(self, q,k,v, mask = None):
        b, n, h, device, bsz, eps = q.size(0), q.size(-2), self.heads, q.device, self.block_size, self.eps

        # pad sequence length to power of 2

        pad_to_len = 2 ** ceil(log2(max(q.size(-2),k.size(-2))))
        padding = 2 ** ceil(log2(max(q.size(-2),k.size(-2)))) - max(q.size(-2),k.size(-2))

        if bool((2 ** ceil(log2(q.size(-2))) - q.size(-2)) > 0) or bool( ( ( 2 ** ceil( log2(k.size(-2)) ) ) - k.size(-2)) > 0 ):
            q,k,v = map(lambda x: F.pad(x, (0, 0, 0, (2 ** ceil(log2(x.size(-2))) - x.size(-2))), value = 0), (q, k, v))

            if exists(mask):
                mask = F.pad(mask, (0, padding), value = False)

        # split out heads, and also divide sequence into blocks

        q, k, v = map(lambda t: rearrange(t, 'b h n d -> (b h) n d'), (q, k, v))

        # scale

        q = q * (q.size(-1)** -0.5)

        # calculate number of levels until 2 x 2

        num_levels = int(log2(pad_to_len // bsz)) - 1

        # coarsening

        qkvs = [(q, k, v, mask)]

        for level in range(num_levels):
            if q.size(-2)%2 == 0:
                q = rearrange(q, 'b (n r) d -> b n r d', r = 2)
            if k.size(-2)%2 == 0:
                k = rearrange(k, 'b (n r) d -> b n r d', r = 2)
                v = rearrange(v, 'b (n r) d -> b n r d', r = 2)

            if exists(mask):
                mask = rearrange(mask, 'b (n r) -> b n r', r = 2)

            # masked mean for queries and keys, but not values

            q = masked_aggregate(q, mask, dim = 2)
            k = masked_aggregate(k, mask, dim = 2)
            v = masked_aggregate(v, mask, dim = 2, average = False)

            if exists(mask):
                mask = torch.any(mask, dim = 2)

            coarsened_qkvs = (q, k, v, mask)
            qkvs.append(coarsened_qkvs)

        # half-attention function

        def calculate_Y_and_A(q, k, v, mask = None):
            S = einsum('... i d, ... j d -> ... i j', q, k)

            if exists(mask):
                mask_value = -torch.finfo(S.dtype).max
                S = S.masked_fill(~mask, mask_value)

            S = S - torch.amax(S, dim = -1, keepdim = True)
            A = S.exp()

            y = einsum('... i j, ... j d -> ... i d', A, v)

            A = A.sum(dim = -1)

            y = rearrange(y, 'b ... n d -> b (... n) d')
            A = rearrange(A, 'b ... i -> b (... i)')
            return y, A

        to_blocks = lambda t: rearrange(t, 'b (n z) ... -> b n z ...', z = bsz)

        # calculate Ys, as in the paper

        Ys = []

        for ind, (q, k, v, mask) in enumerate(reversed(qkvs)):
            is_last = ind == (len(qkvs) - 1)

            q, k, v = map(to_blocks, (q, k, v))

            # generate the mask for S

            S_mask = None
            if exists(mask):
                mask = to_blocks(mask)
                q_mask = mask
                k_mask = flip_every_two(mask) if not is_last else mask
                S_mask = rearrange(q_mask, '... n -> ... n ()') * rearrange(k_mask, '... n -> ... () n')

            # flip keys and values to capture the off-diagonals

            if not is_last:
                k, v = map(flip_every_two, (k, v))

            Y_level = calculate_Y_and_A(q, k, v, mask = S_mask)
            Ys.append(Y_level)

        # interpolate

        Y = 0
        A = 0

        for Y_level, A_level in Ys:
            if torch.is_tensor(Y):
                Y = repeat(Y, 'b n d -> b (n r) d', r = 2)

            if torch.is_tensor(A):
                A = repeat(A, 'b n -> b (n r)', r = 2)

            Y = Y_level + Y
            A = A_level + A

        out = Y / rearrange(A + eps, 'b n -> b n ()')

        # combine out

        out = rearrange(out, '(b h) n d -> b h n d', h = h)

        return out[:, :, :n]

class CausalHAttention1D(nn.Module):
    def __init__(
        self,
        max_seq_len=2**17,
        heads = 8,
        block_size = 16,
        eps = 1e-8,
    ):
        super().__init__()
        self.eps = eps
        self.heads = heads
        self.block_size = block_size
        # derive mask
        self.max_seq_len = max_seq_len
        self.generate_mask(max_seq_len)


    def generate_mask(self,max_seq_len=None):
        max_seq_len = default(max_seq_len,self.max_seq_len)

        num_levels = int(log2(max_seq_len // self.block_size)) - 1
        root_seq = torch.arange(max_seq_len)
        seqs = [root_seq]
        seq = root_seq

        for ind in range(num_levels):
            seq = rearrange(seq, '(n r) -> n r', r = 2)
            seq = seq.amax(dim = -1)
            expanded_mask_seq = repeat(seq, 'n -> (n r)', r = (2 ** (ind + 1)))
            seqs.append(expanded_mask_seq)

        seq_keys = torch.stack(seqs, dim = 0)
        mask = seq_keys > rearrange(root_seq, 'n -> () n')
        self.register_buffer('mask', mask)

    def forward(self, q,k,v, **kwargs):
        b, n, h, device, bsz, eps = q.size(0), q.size(-2), self.heads, q.device, self.block_size, self.eps

        # pad sequence length to power of 2

        pad_to_len = 2 ** ceil(log2(n))
        padding = pad_to_len - n

        if pad_to_len > self.max_seq_len:
            self.max_seq_len = pad_to_len
            self.generate_mask(pad_to_len)

        if padding > 0:
            q,k,v = map(lambda x: F.pad(x, (0, 0, 0, (2 ** ceil(log2(x.size(-2))) - x.size(-2))), value = 0), (q, k, v))

            if exists(mask):
                mask = F.pad(mask, (0, padding), value = False)

        # split out heads, and also divide sequence into blocks

        q, k, v = map(lambda t: rearrange(t, 'b h n d -> (b h) n d'), (q, k, v))

        # scale

        q = q * (q.size(-1)** -0.5)

        # calculate number of levels until 2 x 2

        num_levels = int(log2(pad_to_len // bsz)) - 1

        # coarsening

        qkvs = [(q, k, v)]

        for level in range(num_levels):
            q, k, v = map(lambda t: rearrange(t, 'b (n r) d -> b n r d', r = 2), (q, k, v))

            # masked mean for queries and keys, but not values

            q = q.mean(dim = 2)
            k = k.mean(dim = 2)
            v = v.sum(dim = 2)

            coarsened_qkvs = (q, k, v)
            qkvs.append(coarsened_qkvs)

        # half-attention function

        def calculate_Y_and_A(q, k, v, mask_right_off_diagonals = False, causal_mask_diagonal = False):
            if mask_right_off_diagonals:
                q, k, v = map(lambda t: rearrange(t, 'b (n r) ... -> b n r ...', r = 2), (q, k, v))
                q, k, v = map(lambda t: t[:, :, 1], (q, k, v))

            S = einsum('... i d, ... j d -> ... i j', q, k)

            if causal_mask_diagonal:
                causal_mask = torch.ones(*S.shape[-2:], device = S.device).triu(1).bool()
                mask_value = -torch.finfo(S.dtype).max
                causal_mask = rearrange(causal_mask, 'i j -> () () i j')
                S = S.masked_fill(causal_mask, mask_value)

            S = S - torch.amax(S, dim = -1, keepdim = True)
            A = S.exp()

            y = einsum('... i j, ... j d -> ... i d', A, v)

            A = A.sum(dim = -1)

            if mask_right_off_diagonals:
                y, A = map(lambda t: rearrange(t, 'b n ... -> b n () ...'), (y, A))
                y = F.pad(y, (0, 0, 0, 0, 1, 0), value = 0.)
                A = F.pad(A, (0, 0, 1, 0), value = 0.)

            y = rearrange(y, 'b ... d -> b (...) d')
            A = rearrange(A, 'b ... -> b (...)')
            return y, A

        to_blocks = lambda t: rearrange(t, 'b (n z) ... -> b n z ...', z = bsz)

        # calculate Ys, as in the paper

        Ys = []

        for ind, (q, k, v) in enumerate(reversed(qkvs)):
            is_last = ind == (len(qkvs) - 1)

            q, k, v = map(to_blocks, (q, k, v))

            # flip keys and values to capture the off-diagonals

            if not is_last:
                k, v = map(flip_every_two, (k, v))

            Y_level = calculate_Y_and_A(q, k, v, mask_right_off_diagonals = not is_last, causal_mask_diagonal = is_last)
            Ys.append(Y_level)

        # interpolate

        def safe_cat(acc, el, dim = 0):
            if not exists(acc):
                return el
            return torch.cat((el, acc), dim = dim)

        Y = None
        A = None

        for Y_level, A_level in Ys:
            Y_level, A_level = map(lambda t: rearrange(t, '... -> () ...'), (Y_level, A_level))

            if torch.is_tensor(Y):
                Y = repeat(Y, '... n d -> ... (n r) d', r = 2)

            if torch.is_tensor(A):
                A = repeat(A, '... n -> ... (n r)', r = 2)

            Y = safe_cat(Y, Y_level)
            A = safe_cat(A, A_level)

        # create causal mask for Y and A

        causal_mask = self.mask[:(num_levels + 1), :pad_to_len]

        # mask and sum

        Y_causal_mask = rearrange(causal_mask, 'h n -> h () n ()')
        A_causal_mask = rearrange(causal_mask, 'h n -> h () n')

        Y = Y.masked_fill(Y_causal_mask, 0.)
        A = A.masked_fill(A_causal_mask, 0.)

        Y = Y.sum(dim = 0)
        A = A.sum(dim = 0)

        # normalize

        out = Y / rearrange(A + eps, 'b n -> b n ()')

        # merge heads

        out = rearrange(out, '(b h) n d -> b h n d', h = h)

        # combine out

        return out[:, :n]

class NystromAttention(nn.Module):
    def __init__(
        self,
        dim = None,
        dim_head = 64,
        heads = 8,
        num_landmarks = 256,
        pinv_iterations = 6,
        residual = True,
        context = False,
        residual_conv_kernel = 33,
        eps = 1e-8,
        dropout = 0.,
        transpose_heads_n_dims = False,
        conv_in = None,
        conv_out = None,
        use_mask = False,
        talking_head_attn = True,
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = default(pinv_iterations,heads)

        self.talking_head_attn = talking_head_attn

        self.heads = heads
        self.scale = dim_head ** -0.5
        """
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        """

        self.residual = residual
        self.context = context
        self.use_mask = bool(not context and use_mask)
        self.transpose_heads_n_dims = transpose_heads_n_dims
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            if not self.transpose_heads_n_dims:
                conv_in = conv_in if conv_in != None else heads
                conv_out = conv_out if conv_out != None else heads
                groups = math.gcd(conv_out,conv_in)
                self.res_conv = nn.Conv2d(conv_in, conv_out, (kernel_size, 1), padding = (padding, 0), groups = 1, bias = False)
            else:
                conv_in = conv_in if conv_in != None else dim_head
                conv_out = conv_out if conv_out != None else dim_head
                groups = math.gcd(conv_out,conv_in)
                self.res_conv = nn.Conv2d(conv_in, conv_out, (kernel_size, 1), padding = (padding, 0), groups = 1, bias = False)

        if self.talking_head_attn:
            self.talking_head_sim1 = nn.Parameter(torch.rand((self.heads, self.heads)))
            self.talking_head_sim2 = nn.Parameter(torch.rand((self.heads, self.heads)))
            self.talking_head_sim3 = nn.Parameter(torch.rand((self.heads, self.heads)))
            self.talking_head_attn1 = nn.Parameter(torch.rand((self.heads, self.heads)))
            self.talking_head_attn2 = nn.Parameter(torch.rand((self.heads, self.heads)))
            self.talking_head_attn3 = nn.Parameter(torch.rand((self.heads, self.heads)))

    def forward(self, q, k, v, src_mask = None, mask = None, return_attn = False):
        b, _, n, __, h, m, iters, eps = *q.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        factor = 2
        m = max(12,min(self.num_landmarks,int((k.size(-2)**(1/factor) + q.size(-2)**(1/factor))//2)))

        q_shape = q.shape

        # pad so that sequence can be evenly divided into m landmarks

        remainder = n % m
        padding = m - (n % m)
        if bool(q.size(-2)%m > 0) or bool(k.size(-2)%m > 0):
            #x = F.pad(x, (0, 0, padding, 0), value = 0)
            q,k,v = map(lambda x: F.pad(x, (0, 0, (m - (x.size(-2) % m)), 0), value = 0), (q, k, v))

            if exists(mask):
                mask = F.pad(mask, (padding, 0), value = False)

        # derive query, keys, values
        """
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        """
        # set masked positions to 0 in queries, keys, values

        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        #q = q * self.scale

        # generate landmarks by sum reduction, and then calculate mean using the mask

        l = ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l = ceil(q.size(-2) / m))
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l = ceil(k.size(-2) / m))

        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean

        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, '... (n l) -> ... n', 'sum', l = l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        # masked mean (if mask exists)

        q_landmarks /= divisor
        k_landmarks /= divisor

        # similarities

        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = einsum(einops_eq, q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, k)

        if self.talking_head_attn:
           sim1 = einsum('b h i j, h k -> b k i j',sim1,self.talking_head_sim1)
           sim2 = einsum('b h i j, h k -> b k i j',sim2,self.talking_head_sim2)
           sim3 = einsum('b h i j, h k -> b k i j',sim3,self.talking_head_sim3)

        # masking

        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        if self.use_mask and q.size(2)==k.size(2):
            if src_mask == None:
                mask = torch.triu(torch.ones((q.size(2),q.size(2))),diagonal=1)
            else:
                mask = F.pad(src_mask, (padding, 0,padding, 0), value = 0)
                
            mask_value = -torch.finfo(q.dtype).max

            sim1_mask = repeat(reduce(mask,"i (j l) -> i j",'min',l=l).bool(),"i j -> b h i j",b=q.size(0),h=q.size(1)) if len(mask.size()) == 2 else reduce(mask,"b h i (j l) -> b h i j",'min',l=l).bool()
            sim2_mask = repeat(reduce(mask,"(i k) (j l) -> i j",'min',k=l,l=l).bool(),"i j -> b h i j",b=q.size(0),h=q.size(1)) if len(mask.size()) == 2 else reduce(mask,"(i k) (j l) -> i j",'min',k=l,l=l).bool()
            sim3_mask = repeat(reduce(mask,"(i l) j -> i j",'min',l=l).bool(),"i j -> b h i j",b=q.size(0),h=q.size(1)) if len(mask.size()) == 2 else reduce(mask,"b h (i l) j -> b h i j",'min',l=l).bool()

            sim1.masked_fill_(sim1_mask.to(q.device), mask_value)
            sim2.masked_fill_(sim2_mask.to(q.device), mask_value)
            sim3.masked_fill_(sim3_mask.to(q.device), mask_value)

        # eq (15) in the paper and aggregate values

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim = -1), (sim1, sim2, sim3))

        attn2_inv = moore_penrose_iter_pinv(attn2, iters)

        if self.talking_head_attn:
           attn1 = einsum('b h i j, h k -> b k i j',attn1,self.talking_head_attn1)
           attn2_inv = einsum('b h i j, h k -> b k i j',attn2_inv,self.talking_head_attn2)
           attn3 = einsum('b h i j, h k -> b k i j',attn3,self.talking_head_attn3)

        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        # add depth-wise conv residual of values

        if self.residual:
            context = True if q.size(-2) != k.size(-2) else self.context
            inp = v if not context else q
            if not self.transpose_heads_n_dims:
                out = out + self.res_conv(inp)
            else:
                out = out + self.res_conv(inp.transpose(-3,-1)).transpose(-3,-1)

        # merge and combine heads

        #out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        #out = self.to_out(out)
        out = out[:, :, -n:].reshape(q_shape)

        if return_attn:
            #attn = (attn1.to(torch.device('cpu')) @ attn2_inv.to(torch.device('cpu')) @ attn3.to(torch.device('cpu'))).to(torch.device(q.device))
            attn = attn1 @ attn2_inv @ attn3
            return out, attn
        else:
            return out

class Dynamic_Memory(nn.Module):
    def __init__(self,
                    dim,
                    num_mem_static,
                    num_mem_dyn,
                    *args,
                    max_len=None,
                    warmup_num=None,
                    num_mem_buffer=None,
                    attn='performer',
                    nystromer_landmarks=None,
                    heads=8,
                    nb_features=1024,
                    generalized_attention = True, 
                    kernel_fn = nn.Sigmoid(), 
                    no_projection = False, 
                    **kwargs):
        super(Dynamic_Memory, self).__init__()
        self.mem_static = nn.Parameter(torch.randn((num_mem_static,dim)))
        self.mem_dyn_len = num_mem_dyn
        self.register_buffer(
                            name='mem_dyn',
                            tensor=torch.zeros((num_mem_dyn, dim))
                            )
        num_mem_buffer = default(num_mem_buffer,min(num_mem_dyn,num_mem_static,64))
        self.register_buffer(
                            name='mem_buffer',
                            tensor=torch.zeros((num_mem_buffer, dim))
                            )
        self.register_buffer(
                            name='compressed_mem_buffer',
                            tensor=torch.zeros((num_mem_buffer, dim))
                            )
        self.x_to_mem = nn.Sequential(
            Rearrange("... x y -> ... y x"),
            nn.Conv1d(dim,dim,12,4,6),
            nn.Conv1d(dim,dim,12,4,6),
            nn.Conv1d(dim,dim,12,4,6),
            nn.Conv1d(dim,dim,12,4,6),
            Rearrange("... x y -> ... y x"),
        )
        self.mem_to_kx = nn.Linear(dim, dim)
        self.mem_to_vx = nn.Linear(dim, dim)
        self.x_to_qx = nn.Linear(dim, dim)

        self.max_mem_dyn_len = default(max_len,num_mem_dyn*4)

        nystromer_landmarks = default(nystromer_landmarks,32)
        self.heads = heads

        if attn == 'nystrom':
            self.attn = NystromAttention(dim=dim,dim_head=dim//heads,heads=heads,num_landmarks=nystromer_landmarks,context=True,transpose_heads_n_dims=False,conv_in=heads,conv_out=heads,use_mask=False)
            self.mem_attn = NystromAttention(dim=dim,dim_head=dim,heads=1,num_landmarks=nystromer_landmarks,context=True,transpose_heads_n_dims=False,conv_in=1,conv_out=1,use_mask=False)
        
        elif attn == 'performer':
            self.attn = FastAttention(dim//heads, nb_features, generalized_attention = generalized_attention, kernel_fn = kernel_fn, no_projection = no_projection)
            self.mem_attn = FastAttention(dim, nb_features, generalized_attention = generalized_attention, kernel_fn = kernel_fn, no_projection = no_projection)
        
        elif 'hierarchical' in attn.lower() or 'h_attn' in attn.lower() or 'hattn' in attn.lower():
            self.attn = HAttention1D(heads=heads)
            self.mem_attn = HAttention1D(heads=1)
            
        self.x_to_kmem = nn.Linear(dim, dim)
        self.x_to_vmem = nn.Linear(dim, dim)

        self.emb = RotaryEmbedding(dim)

        self.scale_down = nn.Sequential(
            Rearrange("... x y -> ... y x"),
            nn.Conv1d(dim,dim,12,4,6),
            nn.Conv1d(dim,dim,12,4,6),
            nn.Conv1d(dim,dim,12,4,6),
            nn.Conv1d(dim,dim,12,4,6),
            Rearrange("... x y -> ... y x"),
        )

        self.sigma = nn.Sigmoid()
        self.num_layers = 2
        self.lstm_hs = dim
        self.bidirectional = True
        self._p = nn.LSTM(dim,self.lstm_hs,self.num_layers,batch_first=True,bidirectional=self.bidirectional,proj_size=1)
        self.p = nn.Linear(2,1)

        self.concentration_threshold = 0.8
        self.barely_filled_threshold = 0.2

        self.norm_over_batch = nn.LSTM(dim,dim,batch_first=True,bidirectional=True)
        self.norm_over_batch_lin = nn.Linear(dim*2, dim)

        self.norm = RMSNorm(dim)
        self.gru = nn.GRUCell(dim,dim)
        self.mogrifier = Mogrifier(dim, iters = 13, factorize_k = dim//4)

        self.init_counter = 0
        self.warmup_num = default(warmup_num,8192)

    def hidden_state_set(self,values):
        self.mem_dyn = values['mem_dyn']
        self.mem_buffer = values['mem_buffer']
        self.compressed_mem_buffer = values['compressed_mem_buffer']

    def hidden_state_get(self):
        return {'mem_dyn':self.mem_dyn,'mem_buffer':self.mem_buffer,'compressed_mem_buffer':self.compressed_mem_buffer}
    
    def forward(self,x):
            
        src = x.clone()

        x = self.x_to_qx(self.norm(x))
        
        kv = torch.cat((self.compressed_mem_buffer,self.mem_buffer,self.mem_static,self.mem_dyn),dim=-2).unsqueeze(0)
        k,v = self.mem_to_kx(kv),self.mem_to_vx(kv)

        pos_emb_x = self.emb(x)
        pos_emb_k = self.emb(k)

        if exists(pos_emb_x):
            x = apply_rotary_emb(x, pos_emb_x)
            k = apply_rotary_emb(k, pos_emb_k)

        x,k,v = map(lambda t: rearrange(t,"b n (h d) -> b h n d", h = self.heads,d = x.size(-1)//self.heads),(x,k,v))
        
        k,v = k.repeat(x.size(0),1,1,1),v.repeat(x.size(0),1,1,1)
        x = ckpt(self.attn,x,k,v)

        p,_ = self._p(self.mem_dyn.unsqueeze(0))
        p = self.sigma(ckpt(self.p,p)).squeeze(0)

        prob_aggr = reduce(p,"n d -> d",'mean')
        mem_dyn = self.mem_dyn.unsqueeze(0).unsqueeze(0)

        x = rearrange(x,"b h n d -> b n (h d)")

        if int(prob_aggr) > self.concentration_threshold  and  mem_dyn.size(-2)+(x.size(-2)/256) <= self.max_mem_dyn_len:
            if self.init_counter > self.warmup_num:
                tokens = ckpt(self.scale_down,x).reshape(1,-1,x.size(-1))
                tokens, _ = ckpt(self.norm_over_batch,tokens)
                tokens = self.norm_over_batch_lin(tokens)

                tokens = tokens.reshape(x.size(0),-1,x.size(-1))
                tokens = reduce(tokens,"b n d -> n d",'mean').unsqueeze(0).unsqueeze(0)
                mem_dyn = torch.cat((mem_dyn,tokens),dim=-2)
            
            self.init_counter += 1

        x,src = self.mogrifier(x,src)

        x_shape = x.shape

        x = ckpt(self.gru,x.reshape(-1,x.size(-1)),src.reshape(-1,src.size(-1))).reshape(x_shape)

        xk = self.x_to_kmem(x).unsqueeze(1)
        xv = self.x_to_vmem(x).unsqueeze(1)
        mem_dyn = mem_dyn.repeat(x.size(0),1,1,1)
        mem_dyn = ckpt(self.mem_attn,mem_dyn,xk,xv)
        self.mem_dyn = reduce(mem_dyn,"b h n d -> n d",'mean').clone().detach()

        mem_buffer = torch.cat((self.mem_buffer,reduce(ckpt(self.x_to_mem,x),"b n d -> n d",'mean')),dim=-2)
        self.mem_buffer = mem_buffer[-self.mem_buffer.size(-2):].clone().detach()

        self.compressed_mem_buffer = torch.cat((self.compressed_mem_buffer,ckpt(self.x_to_mem,torch.cat((self.compressed_mem_buffer.unsqueeze(0),self.mem_buffer.unsqueeze(0)),dim=-2)).squeeze(0)),dim=-2)[-self.compressed_mem_buffer.size(-2):].clone().detach()

        return (x, p,self.mem_dyn.size(-2),((abs(self.mem_dyn.size(-2)**2 - self.mem_dyn_len**2)**0.5) / self.mem_dyn_len))
    
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        heads = 16,
        dim_head = None,
        local_heads = 0,
        local_window_size = 512,
        nystromer_landmarks = None,
        nb_features = 1024,
        feature_redraw_interval = 1024,
        generalized_attention = True,
        kernel_fn = nn.Sigmoid(),
        dropout = 0.25,
        no_projection = False,
        qkv_bias = False,
        attn_out_bias = True,
        max_seq_len = 2**17,
        pos_scales = 0,
        content_rel_attn = True,
        rotary_pos_emb = True,
        fixed_emb = False,
        axial_position_emb = False,
        axial_position_shape = None,
        num_mem_kv = 128,
        num_prev_state = None,
        to_q = None,
        to_k = None,
        to_v = None,
        to_out = None,
        hop_attn = None,
        attn = 'performer',
        attend_to_self = False,
        context = True,
        use_mask = False,
        num_prev_mem = None,
    ):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        self.pos_scales = pos_scales = int(default(pos_scales,2 * math.log(2,dim)))
        #dim_head += 2*pos_scales
        additional_head_dims = 2*pos_scales if not content_rel_attn else 4*pos_scales

        self.pass_mask = bool(attn == 'nystrom')

        self.heads = heads
        self.global_heads = global_heads = heads - local_heads

        nystromer_landmarks = default(nystromer_landmarks,512)
        if attn == 'nystrom':
            conv_in = dim_head + additional_head_dims if pos_scales and pos_scales!=None else global_heads
            conv_out = dim_headif if pos_scales and pos_scales!=None else global_heads
            self.fast_attention = NystromAttention(dim=dim,dim_head=dim_head + additional_head_dims,heads=global_heads,num_landmarks=nystromer_landmarks,context=context,transpose_heads_n_dims=bool(pos_scales and pos_scales!=None),conv_in=conv_in,conv_out=conv_out,use_mask=use_mask)
        elif attn == 'performer':
            self.fast_attention = FastAttention(dim_head + additional_head_dims, nb_features, causal = causal, generalized_attention = generalized_attention, kernel_fn = kernel_fn, no_projection = no_projection)
        elif 'hierarchical' in attn.lower() or 'h_attn' in attn.lower() or 'hattn' in attn.lower():
            self.fast_attention = HAttention1D(heads=global_heads) if bool((not causal) and (not use_mask)) else CausalHAttention1D(heads=global_heads)
        self.local_attn = LocalAttention(window_size = local_window_size, causal = bool(use_mask or causal), autopad = True, dropout = dropout, look_forward = int(not bool(use_mask or causal)), rel_pos_emb_config = (dim_head, local_heads)) if local_heads > 0 else None

        self.fixed_emb = fixed_emb
        if rotary_pos_emb:
            self.pos_emb = None
            self.layer_pos_emb = FixedPositionalEmbedding() if fixed_emb else RotaryEmbedding(dim)
        elif axial_position_emb:
            axial_position_shape = default(axial_position_shape, (math.ceil(max_seq_len / 64), 64))
            self.pos_emb = AxialPositionalEmbedding(dim, axial_position_shape)
            self.layer_pos_emb = None
        else:
            self.pos_emb = None
            self.layer_pos_emb = None

        self.to_q = default(to_q,nn.Linear(dim, inner_dim, bias = qkv_bias))
        self.to_k = default(to_k,nn.Linear(dim, inner_dim, bias = qkv_bias))
        self.to_v = default(to_v,nn.Linear(dim, inner_dim, bias = qkv_bias))
        self.to_out = default(to_out,nn.Linear(inner_dim, dim, bias = attn_out_bias))
        self.dropout = nn.Dropout(dropout)

        if pos_scales>0:
            self.q_rel_pos_emb = ConstrainedLinear(
                self.global_heads*dim_head,
                self.global_heads*dim_head,
                pos_scales,
                self.global_heads,
                content_rel_attn=content_rel_attn
            )
            self.k_rel_pos_emb = IdentityLinear(
                self.global_heads*dim_head,
                self.global_heads*dim_head,
                pos_scales,
                self.global_heads,
                content_rel_attn=content_rel_attn
            )
            self.rel_pos_emb_q = LearnableSinusoidEncoding(pos_scales*2)
            self.rel_pos_emb_k = LearnableSinusoidEncoding(pos_scales*2)

        self.attn_to_self = None
        if attend_to_self:
            self_head_dim = 1
            self.features = 17
            scale = 2
            self.feat_prep = nn.Conv1d(inner_dim,inner_dim*scale,kernel_size=self.features,padding=self.features//2,padding_mode="replicate",groups=1)
            self.project_down = nn.Linear(inner_dim*scale, inner_dim)

            if attn == 'nystrom':
                self.attn_to_self = NystromAttention(dim=dim,dim_head=self_head_dim,heads=1,num_landmarks=2**(int(math.log(2,math.log(2,inner_dim*scale)))) , transpose_heads_n_dims=True,talking_head_attn=False)
            elif attn == 'performer':
                self.attn_to_self = FastAttention(self_head_dim, nb_features, causal = False, generalized_attention = generalized_attention, kernel_fn = kernel_fn, no_projection = no_projection)
            elif 'hierarchical' in attn.lower() or 'h_attn' in attn.lower() or 'hattn' in attn.lower():
                self.attn_to_self = HAttention1D(heads=1)
        self.num_mem_kv = num_mem_kv
        num_prev_state = default(num_prev_state,num_mem_kv)
        num_prev_mem = default(num_prev_mem,min(dim,num_mem_kv))
        
        attn = 'performer'

        if num_mem_kv > 0:
            self.mem_k = nn.Parameter(torch.randn(self.heads, num_mem_kv, dim_head))
            self.mem_v = nn.Parameter(torch.randn(self.heads, num_mem_kv, dim_head))
            self.register_buffer(
                                name='prev_state',
                                tensor=torch.zeros((self.heads, num_prev_state, dim_head))
                                )
            
            self.register_buffer(
                                name='prev_mem',
                                tensor=torch.zeros((self.heads, num_prev_mem, dim_head))
                                )
            if num_prev_mem > 0:
                self.scale_down = nn.Sequential(
                    Rearrange("... x y -> ... y x"),
                    nn.Conv1d(dim,dim,12,4,6),
                    nn.Conv1d(dim,dim,12,4,6),
                    nn.Conv1d(dim,dim,12,4,6),
                    nn.Conv1d(dim,dim,12,4,6),
                    Rearrange("... x y -> ... y x"),
                )
            else:
                self.scale_down = nn.Identity()
            #self.register_buffer(name='prev_state_supplementary',tensor=torch.eye(dim_head*self.heads))
            #self.prev_state_parameter = nn.Parameter(torch.randn((dim_head*self.heads,dim_head*self.heads)))
            self.hop_attn = hop_attn
            self.mem_lin_k = nn.Linear(dim_head,dim_head)
            self.mem_lin_v = nn.Linear(dim_head,dim_head)
            self.out_mem = nn.Linear(dim_head,dim_head)
            
            if attn == 'nystrom':
                conv_in = dim_head + additional_head_dims if pos_scales and pos_scales!=None else heads
                conv_out = dim_headif if pos_scales and pos_scales!=None else heads
                self.mem_attn = NystromAttention(dim=dim,dim_head=dim_head + additional_head_dims,context=True,heads=self.heads,num_landmarks=nystromer_landmarks,transpose_heads_n_dims=bool(pos_scales and pos_scales!=None),conv_in=conv_in,conv_out=conv_out)
                self.prev_state_attn = NystromAttention(dim=dim,dim_head=dim_head,context=True,heads=self.heads,num_landmarks=nystromer_landmarks)
            elif attn == 'performer':
                self.mem_attn = FastAttention(dim_head + additional_head_dims, nb_features, causal = False, generalized_attention = generalized_attention, kernel_fn = kernel_fn, no_projection = no_projection)
                self.prev_state_attn = FastAttention(dim_head, nb_features, causal = False, generalized_attention = generalized_attention, kernel_fn = kernel_fn, no_projection = no_projection)
            elif 'hierarchical' in attn.lower() or 'h_attn' in attn.lower() or 'hattn' in attn.lower():
                self.mem_attn = HAttention1D(heads=self.heads)
                self.prev_state_attn = HAttention1D(heads=self.heads)
                
            self.out_k = nn.Linear(dim_head,dim_head)
            self.out_v = nn.Linear(dim_head,dim_head)

            if pos_scales>0:
                self.q_rel_pos_emb_mem = ConstrainedLinear(
                    self.heads*dim_head,
                    self.heads*dim_head,
                    pos_scales,
                    self.heads,
                    content_rel_attn=content_rel_attn
                )
                self.k_rel_pos_emb_mem = IdentityLinear(
                    self.heads*dim_head,
                    self.heads*dim_head,
                    pos_scales,
                    self.heads,
                    content_rel_attn=content_rel_attn
                )
                self.rel_pos_emb_q_mem = LearnableSinusoidEncoding(pos_scales*2)
                self.rel_pos_emb_k_mem = LearnableSinusoidEncoding(pos_scales*2)
        else:
            self.prev_state = None
            self.prev_mem = None

    def hidden_state_set(self,values):
        self.prev_state = values['prev_state']
        self.prev_emm = values['prev_mem']

    def hidden_state_get(self):
        return {'prev_state':self.prev_state,'prev_mem':self.prev_mem}

    def forward(self, x, context = None, src_mask = None, mask = None, context_mask = None, **kwargs):
        b, n, d, h, gh = *x.shape, self.heads, self.global_heads
        
        cross_attend = exists(context)

        if exists(self.pos_emb):
            x = x + self.pos_emb(x)

        context = default(context, x)
        context_mask = default(context_mask, mask) if not cross_attend else context_mask

        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

        pos_emb_q = self.layer_pos_emb(q) if exists(self.layer_pos_emb) else None
        pos_emb_k = self.layer_pos_emb(k) if exists(self.layer_pos_emb) else None

        if self.attn_to_self != None:
            self_q = q
            self_q = ckpt(self.feat_prep,self_q.transpose(-1,-2)).transpose(-1,-2)
            org_shape = self_q.shape
            self_q = rearrange(self_q, 'b n d -> b n d 1')
            self_q = ckpt(self.attn_to_self,self_q,self_q,self_q)
            self_q = rearrange(self_q, 'b n d 1 -> b n d').reshape(org_shape)
            q = ckpt(self.project_down,self_q)
            del(self_q)

        if exists(pos_emb_q):
            q = apply_rotary_emb(q, pos_emb_q)
            k = apply_rotary_emb(k, pos_emb_k)

        tmp_k,tmp_v = k,v

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))

        attn_outs = []

        if not empty(q):
            if exists(context_mask):
                global_mask = context_mask[:, None, :, None]
                v.masked_fill_(~global_mask, 0.)

            if self.pos_scales>0:

                pos_q = rearrange(self.rel_pos_emb_q(torch.arange(q.size(-2), dtype=torch.float32,device=x.device)[None, :, None]),'b n p d -> b n (p d)')
                pos_k = rearrange(self.rel_pos_emb_k(torch.arange(k.size(-2), dtype=torch.float32,device=x.device)[None, :, None]),'b n p d -> b n (p d)')
                
                q = ckpt(self.q_rel_pos_emb,q,pos_q)
                k = ckpt(self.k_rel_pos_emb,k,pos_k)

            out = ckpt(self.fast_attention,q, k, v, src_mask) if self.pass_mask else ckpt(self.fast_attention,q, k, v)
            attn_outs.append(out)

        if not empty(lq):
            assert not cross_attend, 'local attention is not compatible with cross attention'
            out = ckpt(self.local_attn,lq, lk, lv, mask)
            attn_outs.append(out)

        out = torch.cat(attn_outs, dim = 1)

        if self.num_mem_kv > 0:
            mem_k, mem_v, prev_state, prev_mem = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), (self.mem_k, self.mem_v, self.prev_state,self.prev_mem))

            if exists(self.hop_attn):
                hop_k = ckpt(self.hop_attn,tmp_k).reshape(b,self.heads,-1,d//h)
                hop_v = ckpt(self.hop_attn,tmp_v).reshape(b,self.heads,-1,d//h)
                mem_k = torch.cat((mem_k,hop_k),dim=-2)
                mem_v = torch.cat((mem_v,hop_v),dim=-2)

            mem_k = torch.cat((mem_k,prev_state,prev_mem),dim=-2).requires_grad_(True)
            mem_v = torch.cat((mem_v,prev_state,prev_mem),dim=-2).requires_grad_(True)
            mem_k = self.mem_lin_k(mem_k)
            mem_v = self.mem_lin_v(mem_v) 

            if exists(self.pos_emb):
                mem_k = mem_k + self.pos_emb(mem_k.reshape(b,-1,d)).reshape(b,h,-1,d//h)

            if exists(self.layer_pos_emb):
                mem_pos_emb_mem_k = self.layer_pos_emb(mem_k.reshape(b,-1,d))
            else:
                mem_pos_emb_mem_k = None

            if exists(mem_pos_emb_mem_k):
                mem_k = apply_rotary_emb(mem_k.reshape(b,-1,d), mem_pos_emb_mem_k).reshape(b,h,-1,d//h)

            if self.pos_scales>0:
                mem_pos_q = rearrange(self.rel_pos_emb_q_mem(torch.arange(out.size(-2), dtype=torch.float32,device=x.device)[None, :, None]),'b n p d -> b n (p d)')
                mem_pos_k = rearrange(self.rel_pos_emb_k_mem(torch.arange(mem_k.size(-2), dtype=torch.float32,device=x.device)[None, :, None]),'b n p d -> b n (p d)')
                
                out = ckpt(self.q_rel_pos_emb_mem,out,mem_pos_q)
                mem_k = ckpt(self.k_rel_pos_emb_mem,mem_k,mem_pos_k)

            out = ckpt(self.mem_attn,out,mem_k,mem_v)
            out = ckpt(self.out_mem,out)
            self.prev_mem = reduce(torch.cat((prev_mem,self.scale_down(torch.cat((prev_mem,out),dim=-2))),dim=-2)[:,:,self.prev_mem.size(-2)],'b h n d -> h n d','mean').reshape(self.prev_mem.shape)

            out_k = self.out_k(out)
            out_v = self.out_v(out)
            #mat = torch.einsum("b n d, b n e -> b d e",rearrange(out_k,"b h n d -> b n (h d)"),rearrange(out_v,"b h n d -> b n (h d)"))
            #mat = torch.einsum("... i j, ... j k, b ... k l -> b i l",self.prev_state_supplementary,self.prev_state_parameter,mat)
            #prev_state = rearrange(torch.einsum("b n d, b d k -> b n k",rearrange(prev_state,"b h n d -> b n (h d)"),mat),"b n (h d) -> b h n d",h=h,d=d//h)
            prev_state = ckpt(self.prev_state_attn,prev_state,out_k,out_v)
            self.prev_state = reduce(prev_state,"b h n d -> h n d",'mean').reshape(self.prev_state.shape)
            #self.prev_state_supplementary = reduce(mat,"b x y -> x y",'mean')


        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        torch.cuda.empty_cache()
        out = self.dropout(out)

        return out

class SelfAttention(Attention):
    def forward(self, *args, context = None, **kwargs):
        assert not exists(context), 'self attention should not receive context'
        return super().forward(*args, **kwargs)

class CrossAttention(Attention):
    def forward(self, *args, context = None, **kwargs):
        assert exists(context), 'cross attention should receive context'
        return super().forward(*args, context = context, **kwargs)

# performer

class Performer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        local_attn_heads = 0,
        local_window_size = 256,
        causal = False,
        ff_mult = 4,
        nb_features = None,
        feature_redraw_interval = 1000,
        reversible = False,
        ff_chunks = 1,
        generalized_attention = False,
        kernel_fn = nn.ReLU(),
        use_scalenorm = False,
        use_rezero = False,
        ff_glu = False,
        ff_dropout = 0.,
        attn_dropout = 0.,
        cross_attend = False,
        no_projection = False,
        auto_check_redraw = True,
        qkv_bias = True,
        attn_out_bias = True
    ):
        super().__init__()
        layers = nn.ModuleList([])
        local_attn_heads = cast_tuple(local_attn_heads)
        local_attn_heads = local_attn_heads * depth if len(local_attn_heads) == 1 else local_attn_heads
        assert len(local_attn_heads) == depth, 'tuple specifying number of local attention heads per depth must be equal to the total depth'
        assert all(map(lambda n: n >= 0 and n <= heads, local_attn_heads)), 'local attention head value must be less than the total number of heads'

        if use_scalenorm:
            wrapper_fn = partial(PreScaleNorm, dim)
        elif use_rezero:
            wrapper_fn = ReZero
        else:
            wrapper_fn = partial(PreLayerNorm, dim)

        for _, local_heads in zip(range(depth), local_attn_heads):
            layers.append(nn.ModuleList([
                wrapper_fn(SelfAttention(dim, causal = causal, heads = heads, dim_head = dim_head, local_heads = local_heads, local_window_size = local_window_size, nb_features = nb_features, generalized_attention = generalized_attention, kernel_fn = kernel_fn, dropout = attn_dropout, no_projection = no_projection, qkv_bias = qkv_bias, attn_out_bias = attn_out_bias)),
                wrapper_fn(Chunk(ff_chunks, FeedForward(dim, mult = ff_mult, dropout = ff_dropout, glu = ff_glu), along_dim = 1))
            ]))

            if not cross_attend:
                continue

            layers.append(nn.ModuleList([
                wrapper_fn(CrossAttention(dim, heads = heads, dim_head = dim_head, nb_features = nb_features, generalized_attention = generalized_attention, kernel_fn = kernel_fn, dropout = attn_dropout, no_projection = no_projection, qkv_bias = qkv_bias, attn_out_bias = attn_out_bias)),
                wrapper_fn(Chunk(ff_chunks, FeedForward(dim, mult = ff_mult, dropout = ff_dropout, glu = ff_glu), along_dim = 1))
            ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence

        route_attn = ((True, False),) * depth * (2 if cross_attend else 1)
        route_context = ((False, False), (True, False)) * depth
        attn_route_map = {'mask': route_attn, 'pos_emb': route_attn}
        context_route_map = {'context': route_context, 'context_mask': route_context} if cross_attend else {}
        self.net = execute_type(layers, args_route = {**attn_route_map, **context_route_map})

        # keeping track of when to redraw projections for all attention layers
        self.auto_check_redraw = auto_check_redraw
        self.proj_updater = ProjectionUpdater(self.net, feature_redraw_interval)

    def fix_projection_matrices_(self):
        self.proj_updater.feature_redraw_interval = None

    def forward(self, x, **kwargs):
        if self.auto_check_redraw:
            self.proj_updater.redraw_projections()
        return self.net(x, **kwargs)

class PerformerLM(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        dim,
        depth,
        heads,
        dim_head = 64,
        local_attn_heads = 0,
        local_window_size = 256,
        causal = False,
        ff_mult = 4,
        nb_features = None,
        feature_redraw_interval = 1000,
        reversible = False,
        ff_chunks = 1,
        ff_glu = False,
        emb_dropout = 0.,
        ff_dropout = 0.,
        attn_dropout = 0.,
        generalized_attention = False,
        kernel_fn = nn.ReLU(),
        use_scalenorm = False,
        use_rezero = False,
        cross_attend = False,
        no_projection = False,
        tie_embed = False,
        rotary_position_emb = True,
        axial_position_emb = False,
        axial_position_shape = None,
        auto_check_redraw = True,
        qkv_bias = False,
        attn_out_bias = False
    ):
        super().__init__()
        local_attn_heads = cast_tuple(local_attn_heads)

        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, dim)

        if rotary_position_emb:
            self.pos_emb = FixedPositionalEmbedding(dim, max_seq_len)
            self.layer_pos_emb = FixedPositionalEmbedding(dim_head, max_seq_len)
        elif axial_position_emb:
            axial_position_shape = default(axial_position_shape, (math.ceil(max_seq_len / 64), 64))
            self.pos_emb = AxialPositionalEmbedding(dim, axial_position_shape)
            self.layer_pos_emb = Always(None)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len)
            self.layer_pos_emb = Always(None)

        self.dropout = nn.Dropout(emb_dropout)

        self.performer = Performer(dim, depth, heads, dim_head, local_attn_heads, local_window_size, causal, ff_mult, nb_features, feature_redraw_interval, reversible, ff_chunks, generalized_attention, kernel_fn, use_scalenorm, use_rezero, ff_glu, ff_dropout, attn_dropout, cross_attend, no_projection, auto_check_redraw, qkv_bias, attn_out_bias)
        self.norm = ScaleNorm(dim)
        self.to_out = nn.Linear(dim, num_tokens) if not tie_embed else None

    def check_redraw_projections(self):
        self.performer.check_redraw_projections()

    def fix_projection_matrices_(self):
        self.performer.fix_projection_matrices_()

    def forward(self, x, return_encodings = False, **kwargs):
        b, n, device = *x.shape, x.device
        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'

        # token and positional embeddings
        x = self.token_emb(x)
        x += self.pos_emb(x)

        x = self.dropout(x)

        # performer layers

        layer_pos_emb = self.layer_pos_emb(x)
        x = self.performer(x, pos_emb = layer_pos_emb, **kwargs)

        # norm and to logits
        x = self.norm(x)

        if return_encodings:
            return x

        if exists(self.to_out):
            return self.to_out(x)

        return x @ self.token_emb.weight.t()
