import  torch
from    torch import nn, einsum

from    einops import rearrange, repeat
from    einops.layers.torch import Rearrange, Reduce
import  torch.nn.functional as F
from    ops.layernorm import LayerNorm2d
from torch.nn import init


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# helper classes

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class Conv_PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 2, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Conv_FeedForward(nn.Module):
    def __init__(self, dim, mult = 2, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Conv2d(dim, inner_dim,1,1,0),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(inner_dim, dim,1,1,0),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)


class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x

class Dropsample(nn.Module):
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob
  
    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device = device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)

def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        # nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride = stride, padding = 1, groups = hidden_dim),
        # nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate = shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        # nn.BatchNorm2d(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout = dropout)


    return net

# attention related classes





class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        padding = kernel_size // 2
        #self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=2, padding=padding, groups=in_channels)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class PVTV2_MB4Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if  linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.dw=DepthwiseSeparableConv(dim,dim)

            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()

    def forward(self, x):

        batch, height, width, window_height, window_width, _= x.shape

        B, N, C = batch*height*width,window_width*window_height,_
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if  self.linear:
            if self.sr_ratio > 1:
                x_ = rearrange(x, 'b x y w1 w2 d -> (b x y) d w1 w2')
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)


        else:
            x_ = rearrange(x, 'b x y w1 w2 d -> (b x y) d w1 w2')
            x_=self.dw(x_).reshape(B, C, -1).permute(0, 2, 1)

            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = rearrange(x, '(b x y) (i j) c -> b x y i j c', x=height, y=width, i=window_width)

        return x

class ds_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
                 dilation=[1, 3, 5], groups=1, bias=True,
                 act_layer='nn.SiLU(True)'):
        super().__init__()
        assert in_planes % groups == 0
        assert kernel_size == 3, 'only support kernel size 3 now'
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.with_bias = bias

        self.weight = nn.Parameter(torch.randn(out_planes, in_planes // groups, kernel_size, kernel_size),
                                   requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_planes))
        else:
            self.bias = None
        self.act = eval(act_layer)



    def forward(self, x):
        output = 0
        for dil in self.dilation:
            output += self.act(
                F.conv2d(
                    x, weight=self.weight, bias=self.bias, stride=self.stride, padding=dil,
                    dilation=dil, groups=self.groups,
                )
            )
        return output

class SE(nn.Module):
    def __init__(self,
                 inp,
                 oup,
                 expansion=0.25):
        """
        Args:
            inp: input features dimension.
            oup: output features dimension.
            expansion: expansion ratio.
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SCATAttention(nn.Module):
    def __init__(self, dim, proj_kernel=3, kv_proj_stride=2, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        padding = proj_kernel // 2
        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim)
        self.dw=DepthwiseSeparableConv(dim,dim)

        # self.to_q = DepthwiseSeparableConv(dim, inner_dim, kernel_size=3, padding = padding, stride = 1,)
        self.to_q = DepthwiseSeparableConv(dim,dim)
        # self.to_kv = SepConv2d(dim, inner_dim * 2, kernel_size=3, padding = padding, stride = 2, )
        #self.to_kv = DepthwiseSeparableConv(dim, inner_dim * 2, stride=2)
        self.to_kv = DepthwiseSeparableConv(dim, dim)
        self.attn_drop = nn.Dropout(0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

        self.cbam=CustomLayer(dim)
        self.conv = nn.Conv2d(dim*2, dim, 1)
    def forward(self, x):
        batch, height, width, window_height, window_width, _ = x.shape

        '''print("x:",x.shape)
        # x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')
        B, H, W, C = batch * height * width, window_width, window_height, _
        #x_ = rearrange(x, 'b x y w1 w2 d -> (b x y) d w1 w2 ')
        x_=x.reshape(B, H * W, self.heads, C // self.heads).permute(0, 2, 1, 3)
        x_ = x_.permute(0, 1, 3, 2).reshape(B, self.dim, H, W).contiguous()

        print("x_:",x_.shape)

        q = self.to_q(x_).reshape(B, H * W, self.heads, C // self.heads).permute(0, 2, 1, 3)
        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')
        print(x.shape)
        B, N, C = batch * height * width, window_width * window_height, _


        q = self.q(x).reshape(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)
        print("q",q.shape)'''
        B, N, C = batch * height * width, window_width * window_height, _
        B, H, W, C = batch * height * width, window_width, window_height, _
        # b=b, n=c, _=h, y=w b=batch,n=d,y=
        # b, n, _, y, h = *shape, self.heads

        #      Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1=w, w2=w)
        # = rearrange(x, 'b x y w1 w2 d -> (b x y) d w1 w2')
        # x_ = self.dw(x_).reshape(B, C, -1).permute(0, 2, 1)

        x_left=self.cbam(x)

        q = self.q(x).reshape(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)

        print("q的shape",q.shape)
        #q = q.reshape(B, self.heads, self.dim // self.heads, H * W,)#.permute(0, 1, 3, 2).contiguous()
        #q = rearrange(q, 'b (a d) h w -> b a (h w) d', b=B, a=self.heads)

        x_ = rearrange(x, 'b x y w1 w2 d -> (b x y) d w1 w2')
        x_=self.dw(x_).reshape(B, C, -1).permute(0, 2, 1)
        kv = self.kv(x_).reshape(B, -1, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        print("kv的shape", kv.shape)

        k, v = kv[0], kv[1]
        print("k的shape", k.shape)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        out=torch.cat((x_left, x), dim=0)
        out=self.conv(out)
        out = rearrange(out, '(b x y) (i j) c -> b x y i j c', x=height, y=width, i=window_width)
        return out


class AgentAttention1(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, agent_num=49, **kwargs):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        # self.num_patches = num_patches
        # window_size = (int(num_patches ** 0.5), int(num_patches ** 0.5))
        # self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.agent_num = agent_num
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
        '''
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window_size[0] // sr_ratio, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window_size[1] // sr_ratio))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], agent_num))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        '''
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch, height, width, window_height, window_width, _ = x.shape
        b, n, c, H, W = batch * height * width, window_width * window_height, _, window_width, window_height
        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')
        b, n, c = x.shape
        num_heads = self.num_heads
        head_dim = c // num_heads
        q = self.q(x)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, H, W)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]

        agent_tokens = self.pool(q.reshape(b, H, W, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        '''
        kv_size = (self.window_size[0] // self.sr_ratio, self.window_size[1] // self.sr_ratio)

        position_bias1 = nn.functional.interpolate(self.an_bias, size=kv_size, mode='bilinear')
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias = position_bias1 + position_bias2 
        '''
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1))
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        '''
        agent_bias1 = nn.functional.interpolate(self.na_bias, size=self.window_size, mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2
        '''

        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1))
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, H // self.sr_ratio, W // self.sr_ratio, c).permute(0, 3, 1, 2)
        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v, size=(H, W), mode='bilinear')
        #x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x)
        x = rearrange(x, '(b x y) (i j) c -> b x y i j c', x=height, y=width, i=window_width)

        return x

class MNV4LayerScale(nn.Module):
    #def __init__(self, inp, init_value):
    def __init__(self, init_value):
        """LayerScale as introduced in CaiT: https://arxiv.org/abs/2103.17239
        Referenced from here https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_blocks.py

        As used in MobileNetV4.

        Attributes:
            init_value (float): value to initialize the diagonal matrix of LayerScale.
        """
        super().__init__()
        self.init_value = init_value
        #self._gamma = nn.Parameter(self.init_value * torch.ones(inp, 1, 1))
        self._gamma=None

    def forward(self, x):
        self._gamma = nn.Parameter(self.init_value * torch.ones(x.size(0), 1, 1, device=x.device))
        return x * self._gamma
class PVTV2_MB4LSAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if  linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.dw=DepthwiseSeparableConv(dim,dim)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.layer_scale_init_value = 1e-5
        self.layer_scale = MNV4LayerScale(self.layer_scale_init_value)

    def forward(self, x):

        batch, height, width, window_height, window_width, _= x.shape

        B, N, C = batch*height*width,window_width*window_height,_
        #x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if  self.linear:
            if self.sr_ratio > 1:
                x_ = rearrange(x, 'b x y w1 w2 d -> (b x y) d w1 w2')
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                print(self.sr_ratio)

        else:
            x_ = rearrange(x, 'b x y w1 w2 d -> (b x y) d w1 w2')
            x_=self.dw(x_).reshape(B, C, -1).permute(0, 2, 1)

            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)



        x=self.layer_scale(x)

        x = rearrange(x, '(b x y) (i j) c -> b x y i j c', x=height, y=width, i=window_width)
        return x


class TransNextAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:  # 这就是使用线性 SRA
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)  # 卷积核为 1
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()  # 激活函数


    def forward(self, x):

        batch, height, width, window_height, window_width, _ = x.shape

        B, N, C = batch * height * width, window_width * window_height, _
        # x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.linear:
            if self.sr_ratio > 1:
                # x_ = x.permute(0, 2, 1).reshape(B, C, window_height, window_width)
                x_ = rearrange(x, 'b x y w1 w2 d -> (b x y) d w1 w2')
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        else:
            x_ = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')
            '''
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            '''

            x_ = x_.permute(0, 2, 1).reshape(B, -1, window_height, window_width).contiguous()
            #x_ = self.pool(self.act(self.sr(x_))).reshape(B, -1, N).permute(0, 2, 1)
            x_ = self.pool(self.act(self.sr(x_))).reshape(B, C,-1).permute(0, 2, 1)

            x_ = self.norm(x_)
            kv_pool = self.kv(x_).reshape(B, -1, 2 ,self.num_heads, C // self.num_heads).permute(2,0, 3, 1, 4)
            #k_pool, v_pool = kv_pool.chunk(2, dim=1)

        k, v = kv_pool[0], kv_pool[1]


        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)


        x = rearrange(x, '(b x y) (i j) c -> b x y i j c', x=height, y=width, i=window_width)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

class Pooling(nn.Module):
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        #x_=rearrange(x, 'b x y w1 w2 d -> (b x y) d w1 w2')
        #return self.pool(x_) - x_
        return self.pool(x)-x
class ChannelMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PoolFormerBlock(nn.Module):
    def __init__(self, dim, pool_size=3, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ChannelMLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Pooling1(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        batch, height, width, window_height, window_width, _ = x.shape
        x_=rearrange(x, 'b x y w1 w2 d -> (b x y) d w1 w2')
        x_=self.pool(x_) - x_
        #x_= rearrange(x_, '(b x y) (i j) c -> b x y i j c', x=height, y=width, i=window_width)
        x_ = rearrange(x_, '(b x y) d w1 w2 -> b x y w1 w2 d ',x=height, y=width, w1=window_width,w2=window_height)
        return x_

class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """
    def __init__(self, dim,
        expansion_ratio=2,
        #expansion_ratio=1,
        #act1_layer=StarReLU,
        act1_layer=nn.ReLU6,
        act2_layer=nn.Identity,
        bias=False, kernel_size=7, padding=3,
        **kwargs, ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size,
            padding=padding, groups=med_channels, bias=bias) # depthwise conv
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x):
        batch, height, width, window_height, window_width, _ = x.shape
        x = rearrange(x, 'b x y w1 w2 d -> (b x y)   w1 w2 d')
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act2(x)
        x = self.pwconv2(x)
        x = rearrange(x, '(b x y)   w1 w2 d -> b x y w1 w2 d ', x=height, y=width, w1=window_width, w2=window_height)
        return x


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias
class SepConvStarReLU(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """
    def __init__(self, dim,
                 expansion_ratio=2,
                 # expansion_ratio=1,
                 act1_layer=StarReLU,
                 # act1_layer=nn.ReLU6,
                 act2_layer=nn.Identity,
                 bias=False, kernel_size=7, padding=3,
                 **kwargs, ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size,
            padding=padding, groups=med_channels, bias=bias)  # depthwise conv
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x):
        batch, height, width, window_height, window_width, _ = x.shape
        x = rearrange(x, 'b x y w1 w2 d -> (b x y)   w1 w2 d')

        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act2(x)
        x = self.pwconv2(x)
        x = rearrange(x, '(b x y)   w1 w2 d -> b x y w1 w2 d ', x=height, y=width, w1=window_width, w2=window_height)
        return x


class HardSwish(nn.Module):
    def forward(self, x):
        return x * torch.nn.functional.relu6(x + 3, inplace=True) / 6


class SqueezeExcite(nn.Module):
    def __init__(self, input_channels, squeeze_factor=4):
        super().__init__()
        squeeze_channels = input_channels // squeeze_factor
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = nn.ReLU(inplace=True)
        self.gate_fn = HardSwish()

    def forward(self, x):
        scale = x.mean((2, 3), keepdim=True)  # Global average pooling
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.gate_fn(scale)
        return x * scale


class Bneck(nn.Module):
    def __init__(self, in_channels, kernel_size=7, stride=1, expand_ratio=2, se_ratio=0.25, act=nn.ReLU,
                 use_hs=True):
        super().__init__()
        out_channels = in_channels
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = stride == 1 and in_channels == out_channels
        self.expand = in_channels != hidden_dim

        if self.expand:
            self.expand_conv = nn.Conv2d(in_channels, hidden_dim, 1, bias=False)
            self.bn0 = nn.BatchNorm2d(hidden_dim)
            self.act0 = HardSwish() if use_hs else act(inplace=True)

        self.depthwise_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride=stride, padding=kernel_size // 2,
                                        groups=hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act1 = HardSwish() if use_hs else act(inplace=True)

        self.se = SqueezeExcite(hidden_dim, squeeze_factor=int(1 / se_ratio)) if se_ratio > 0 else nn.Identity()

        self.project_conv = nn.Conv2d(hidden_dim, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        batch, height, width, window_height, window_width, _ = x.shape
        x = rearrange(x, 'b x y w1 w2 d -> (b x y)  d w1 w2 ')

        identity = x

        if self.expand:
            x = self.expand_conv(x)
            x = self.bn0(x)
            x = self.act0(x)

        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.se(x)

        x = self.project_conv(x)
        x = self.bn2(x)

        if self.use_res_connect:
            return rearrange(x + identity, '(b x y)  d w1 w2  ->b x y w1 w2 d ', x=height, y=width, w1=window_width,
                             w2=window_height)
        else:
            return x


class MobileViTv2Attention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(MobileViTv2Attention, self).__init__()
        self.fc_i = nn.Linear(d_model, 1)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)

        self.d_model = d_model
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :return:
        '''
        i = self.fc_i(input)  # (bs,nq,1)
        weight_i = torch.softmax(i, dim=1)  # bs,nq,1
        context_score = weight_i * self.fc_k(input)  # bs,nq,d_model
        context_vector = torch.sum(context_score, dim=1, keepdim=True)  # bs,1,d_model
        v = self.fc_v(input) * context_vector  # bs,nq,d_model
        out = self.fc_o(v)  # bs,nq,d_model

        return out


class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

class ChannelOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

class CASAttention(nn.Module):
    def __init__(self, dim=512, attn_bias=False, proj_drop=0.):
        super().__init__()
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, stride=1, padding=0, bias=attn_bias)
        self.oper_q = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.oper_k = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch, height, width, window_height, window_width, _ = x.shape
        B, N, C = batch * height * width, window_width * window_height, _
        B, H, W, C = batch * height * width, window_width, window_height, _
        x = rearrange(x, 'b x y w1 w2 d -> (b x y) d w1 w2')

        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)
        out = self.proj(self.dwc(q + k) * v)
        out = self.proj_drop(out)
        out = rearrange(out, '(b x y) c i j  -> b x y i j c', x=height, y=width, i=window_width)

        return out



class newGated_Conv_FeedForward(nn.Module):
    def __init__(self, dim, mult=1, bias=False, dropout=0.):
        super().__init__()

        hidden_features = int(dim * mult)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        # 保留一路 3x3 卷积
        self.dwconv_3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                    groups=hidden_features * 2, bias=bias)

        # 新增一路 5x5 卷积
        self.dwconv_5x5 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2,
                                    groups=hidden_features * 2, bias=bias)

        # 修改 project_out 的输入通道数为拼接后通道数
        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)

        # 通过 3x3 卷积路径
        x1_3, x2_3 = self.dwconv_3x3(x).chunk(2, dim=1)

        # 通过 5x5 卷积路径
        x1_5, x2_5 = self.dwconv_5x5(x).chunk(2, dim=1)

        # 门控机制结合两路输出
        x1 = F.gelu(x1_3) * x2_3  # 3x3 路径
        x2 = F.gelu(x1_5) * x2_5  # 5x5 路径

        # 将两路输出拼接
        x = torch.cat([x1, x2], dim=1)

        # 通过 project_out 恢复通道数
        x = self.project_out(x)
        return x


class LCFFN(nn.Module):
    def __init__(self, dim, mult = 1, bias=False, dropout = 0.):
        super().__init__()

        hidden_features = int(dim * mult)
        self.conv=nn.Conv2d(dim,dim, kernel_size=1, bias=bias)
        self.dwconv=nn.Conv2d(dim, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.out=nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)


    def forward(self, x):
        x=self.dwconv(self.conv(x))
        x1,x2=x.chunk(2,dim=1)
        x=F.gelu(x1)*x2
        x=self.out(x)


        return x




class OSA_Block(nn.Module):
    def __init__(self, channel_num=64, bias = True, ffn_bias=True, window_size=8, with_pe=False, dropout=0.0):
        super(OSA_Block, self).__init__()

        w = window_size

        self.layer = nn.Sequential(
                MBConv(
                    channel_num,
                    channel_num,
                    downsample = False,
                    expansion_rate = 1,
                    shrinkage_rate = 0.25
                ),

                Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1=w, w2=w),

                PreNormResidual(channel_num, PVTV2_MB4Attention(channel_num)),
                Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),
                Conv_PreNormResidual(channel_num, newGated_Conv_FeedForward(dim=channel_num, dropout=dropout)),
                )



    def forward(self, x):

        out = self.layer(x)
        return out
