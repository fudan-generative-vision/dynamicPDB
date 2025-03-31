"""Fork of Openfold's IPA."""

import numpy as np
import torch
import math
from scipy.stats import truncnorm
import torch.nn as nn
from typing import Optional, Callable, List, Sequence
from openfold.utils.rigid_utils import Rigid
from openfold.model.structure_module import AngleResnet
from src.data import all_atom
# from omegafold import config,pipeline
# from model.omega_struceture import StructureModule
import torch.nn.functional as F
import sys
# from ckh_tool.gpu_mem_track import MemTracker


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)

def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f

def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


def compute_angles(ca_pos, pts):
    batch_size, num_res, num_heads, num_pts, _ = pts.shape
    calpha_vecs = (ca_pos[:, :, None, :] - ca_pos[:, None, :, :]) + 1e-10
    calpha_vecs = torch.tile(calpha_vecs[:, :, :, None, None, :], (1, 1, 1, num_heads, num_pts, 1))
    ipa_pts = pts[:, :, None, :, :, :] - torch.tile(ca_pos[:, :, None, None, None, :], (1, 1, num_res, num_heads, num_pts, 1))
    phi_angles = all_atom.calculate_neighbor_angles(
        calpha_vecs.reshape(-1, 3),
        ipa_pts.reshape(-1, 3)
    ).reshape(batch_size, num_res, num_res, num_heads, num_pts)
    return  phi_angles


class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init_fn is not None:
            init_fn(self.weight, self.bias)
        else:
            if init == "default":
                lecun_normal_init_(self.weight)
            elif init == "relu":
                he_normal_init_(self.weight)
            elif init == "glorot":
                glorot_uniform_init_(self.weight)
            elif init == "gating":
                gating_init_(self.weight)
                if bias:
                    with torch.no_grad():
                        self.bias.fill_(1.0)
            elif init == "normal":
                normal_init_(self.weight)
            elif init == "final":
                final_init_(self.weight)
            else:
                raise ValueError("Invalid init string.")


class StructureModuleTransition(nn.Module):
    def __init__(self, c):
        super(StructureModuleTransition, self).__init__()

        self.c = c

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c, self.c, init="final")
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(self.c)

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)
        s = s + s_initial
        s = self.ln(s)

        return s


class EdgeTransition(nn.Module):
    def __init__(
            self,
            *,
            node_embed_size,
            edge_embed_in,
            edge_embed_out,
            num_layers=2,
            node_dilation=2
        ):
        super(EdgeTransition, self).__init__()

        bias_embed_size = node_embed_size // node_dilation
        self.initial_embed = Linear(
            node_embed_size, bias_embed_size, init="relu")
        hidden_size = bias_embed_size * 2 + edge_embed_in
        trunk_layers = []
        for _ in range(num_layers):
            trunk_layers.append(Linear(hidden_size, hidden_size, init="relu"))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)
        self.final_layer = Linear(hidden_size, edge_embed_out, init="final")
        self.layer_norm = nn.LayerNorm(edge_embed_out)

    def forward(self, node_embed, edge_embed):
        node_embed = self.initial_embed(node_embed)
        batch_size, num_res, _ = node_embed.shape
        edge_bias = torch.cat([
            torch.tile(node_embed[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(node_embed[:, None, :, :], (1, num_res, 1, 1)),
        ], axis=-1)
        edge_embed = torch.cat(
            [edge_embed, edge_bias], axis=-1).reshape(
                batch_size * num_res**2, -1)
        edge_embed = self.final_layer(self.trunk(edge_embed) + edge_embed)
        edge_embed = self.layer_norm(edge_embed)
        edge_embed = edge_embed.reshape(
            batch_size, num_res, num_res, -1
        )
        return edge_embed


class InvariantPointAttention(nn.Module):
    """
    Implements Algorithm 22.
    """
    def __init__(
        self,
        ipa_conf,
        inf: float = 1e5,
        eps: float = 1e-8,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super(InvariantPointAttention, self).__init__()
        self._ipa_conf = ipa_conf

        self.c_s = ipa_conf.c_s
        self.c_z = ipa_conf.c_z
        self.c_hidden = ipa_conf.c_hidden
        self.no_heads = ipa_conf.no_heads
        self.no_qk_points = ipa_conf.no_qk_points
        self.no_v_points = ipa_conf.no_v_points
        self.inf = inf
        self.eps = eps

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        self.linear_b = Linear(self.c_z, self.no_heads)
        self.down_z = Linear(self.c_z, self.c_z // 4)

        self.head_weights = nn.Parameter(torch.zeros((ipa_conf.no_heads)))
        ipa_point_weights_init_(self.head_weights)

        # concat_out_dim =  (
        #     self.c_z // 4 + self.c_hidden + self.no_v_points * 4
        # )
        concat_out_dim =  (
            self.c_z // 4 + self.c_hidden + self.no_v_points * 8
        )
        self.linear_out = Linear(self.no_heads * concat_out_dim, self.c_s, init="final")
        # self.linear_out = IPA_MLP(self.no_heads * concat_out_dim, self.c_s)

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()
        # TODO: Remove after published checkpoint is updated without these weights.
        self.linear_rbf = Linear(20, 1)

        # self.temporal = nn.Sequential(
        #     nn.Conv1d(self.no_heads, self.no_heads * 8, kernel_size=3, padding=1, stride=1),
        #     nn.ReLU(True),
        #     nn.Conv1d(self.no_heads * 8, self.no_heads, kernel_size=3, padding=1, stride=1)
        # )

    def forward(
        self,
        s: torch.Tensor,
        z: Optional[torch.Tensor],
        r: Rigid,
        mask: torch.Tensor,
        _offload_inference: bool = False,
        _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        if _offload_inference:
            z = _z_reference_list
        else:
            z = [z]

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [*, N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s)

        # This is kind of clunky, but it's how the original does it
        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].apply(q_pts)

        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
        )

        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)

        # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)

        # [*, N_res, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z[0])
        
        if(_offload_inference):
            z[0] = z[0].cpu()

        # [*, H, N_res, N_res]
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))

        # [*, N_res, N_res, H, P_q, 3]
        pt_displacement = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_displacement ** 2

        # [*, N_res, N_res, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        pt_att = pt_att * head_weights

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))
        
        a = a + pt_att 

        # #print(a.shape)torch.Size([4, 8, 256, 256])
        # #exit()

        # as1, as2, as3, as4 = a.shape
        # a_ = a.permute(2, 3, 1, 0).reshape(as3*as4, as2, as1)
        # a_ = self.temporal(a_)
        # a_ = a_.reshape(as3, as4, as2, as1).permute(3, 2, 0, 1)
        # a  = a + a_

        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)
        # torch.save(a.cpu(),'a.pth')
        # exit()

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(
            a, v.transpose(-2, -3).to(dtype=a.dtype)
        ).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, H, 3, N_res, P_v] 
        o_pt = torch.sum(
            (
                a[..., None, :, :, None]
                * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
            ),
            dim=-2,
        )

        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt_out_ti = o_pt
        # # TODO ref
        # # print(r.to_tensor_7().shape)
        # ref_r = r.to_tensor_7()[:1].repeat(r.shape[0],1,1)
        # # print(ref_r.shape)
        # # print(ref_r[0,1],ref_r[1,1],r.to_tensor_7()[0,1])
        # ref_r = Rigid.from_tensor_7(ref_r)
        # # print(ref_r.shape)
        # # print('='*10)
        # o_pt_out_ti = ref_r[..., None, None].invert_apply(o_pt_out_ti)

        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, N_res, H * P_v]
        o_pt_dists = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(
            o_pt_dists, 2)
        o_pt_dists_out_ti = torch.sqrt(torch.sum(o_pt_out_ti ** 2, dim=-1) + self.eps)
        o_pt_norm_feats_out_ti = flatten_final_dims(o_pt_dists_out_ti, 2)

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)
        o_pt_out_ti = o_pt_out_ti.reshape(*o_pt_out_ti.shape[:-3], -1, 3)

        if(_offload_inference):
            z[0] = z[0].to(o_pt.device)

        # [*, N_res, H, C_z // 4]
        pair_z = self.down_z(z[0]).to(dtype=a.dtype)
        o_pair = torch.matmul(a.transpose(-2, -3), pair_z)

        # [*, N_res, H * C_z // 4]
        o_pair = flatten_final_dims(o_pair, 2)

        o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats, o_pair, *torch.unbind(o_pt_out_ti, dim=-1), o_pt_norm_feats_out_ti]
        # tmp = torch.cat(o_feats, dim=-1).to(dtype=z[0].dtype) # torch.Size([1+Frame, 135, 1344])
        # print(o_feats.shape)
        # exit()

        # [*, N_res, C_s]
        s = self.linear_out(
            torch.cat(
                o_feats, dim=-1
            ).to(dtype=z[0].dtype)
        )
        
        return s


class TorsionAngles(nn.Module):
    def __init__(self, c, num_torsions, eps=1e-8):
        super(TorsionAngles, self).__init__()

        self.c = c
        self.eps = eps
        self.num_torsions = num_torsions

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        # TODO: Remove after published checkpoint is updated without these weights.
        self.linear_3 = Linear(self.c, self.c, init="final")
        self.linear_final = Linear(
            self.c, self.num_torsions * 2, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)

        s = s + s_initial
        unnormalized_s = self.linear_final(s)
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(unnormalized_s ** 2, dim=-1, keepdim=True),
                min=self.eps,
            )
        )
        normalized_s = unnormalized_s / norm_denom

        return unnormalized_s, normalized_s


class ScoreLayer(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out):
        super(ScoreLayer, self).__init__()

        self.linear_1 = Linear(dim_in, dim_hid, init="relu")
        self.linear_2 = Linear(dim_hid, dim_hid)
        self.linear_3 = Linear(dim_hid, dim_out, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = s + s_initial
        s = self.linear_3(s)
        return s


class BackboneUpdate(nn.Module):
    """
    Implements part of Algorithm 23.
    """

    def __init__(self, c_s):
        """
        Args:
            c_s:
                Single representation channel dimension
        """
        super(BackboneUpdate, self).__init__()

        self.c_s = c_s

        self.linear = Linear(self.c_s, 6, init="final")

    def forward(self, s: torch.Tensor):
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 6] update vector 
        """
        # [*, 6]
        update = self.linear(s)

        return update

class TimeBlock(nn.Module):
    def __init__(self, node_dim, time_embed_dim, hidden_dim=None):
        super(TimeBlock,self).__init__()
        self.node_dim = node_dim
        self.time_embed_dim = time_embed_dim
        if hidden_dim is not  None:
            self.hidden_dim = hidden_dim
        else:
            self.hidden_dim = self.node_dim//2
        self.time_proj = nn.Sequential(
                nn.Linear(self.time_embed_dim, 4*self.time_embed_dim),
                nn.SiLU(),
                nn.Linear(4*self.time_embed_dim, self.hidden_dim),
            )
        self.node_proj = nn.Sequential(
            nn.LayerNorm(self.node_dim),
            nn.SiLU(),
            nn.Linear(self.node_dim, self.hidden_dim),
        )
        self.out_prj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.node_dim),
        )
    def forward(self,node_feature,time_embeddings):
        time_feat = self.time_proj(time_embeddings)
        hidden_node_feat = self.node_proj(node_feature)
        node_feat = self.out_prj(time_feat+hidden_node_feat)
        out = node_feature+node_feat
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=40):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)# MAX_L,1
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        # print(pe.shape,#torch.Size([1, 40, 33])
        #       pe[0, :, 0::2].shape,#torch.Size([40, 17])
        #       pe[0, :, 1::2].shape,#torch.Size([40, 16])
        #       position.shape,# torch.Size([40, 1])
        #       div_term.shape#torch.Size([17]
        #       )
        # len_sin = pe[0, :, 0::2].shape[-1]
        # len_cos = pe[0, :, 1::2].shape[-1]

        pe[0, :, 0::2] = torch.sin(position * div_term)#[:len_sin]
        pe[0, :, 1::2] = torch.cos(position * div_term)#[:len_cos]

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ConvNet(nn.Module):
    def __init__(self, dim):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Sequential(
                         nn.Conv2d(dim, dim//2, kernel_size=5, padding=2),
                         nn.ReLU(True),
                         nn.Conv2d(dim//2, dim, kernel_size=5, padding=2),
                         nn.ReLU(True))

        self.conv2 = nn.Sequential(
                         nn.Conv2d(dim, dim//2, kernel_size=5, padding=2),
                         nn.ReLU(True),
                         nn.Conv2d(dim//2, dim, kernel_size=5, padding=2),
                         nn.ReLU(True))

        self.conv3 = nn.Sequential(
                         nn.Conv2d(dim, dim//2, kernel_size=5, padding=2),
                         nn.ReLU(True),
                         nn.Conv2d(dim//2, dim, kernel_size=5, padding=2),
                         nn.ReLU(True))

        self.conv4 = nn.Sequential(
                         nn.Conv2d(dim, dim//2, kernel_size=5, padding=2),
                         nn.ReLU(True),
                         nn.Conv2d(dim//2, dim, kernel_size=5, padding=2),
                         nn.ReLU(True))

    def forward(self, x):

        x = x.permute(2, 0, 1).unsqueeze(0)

        x = self.conv1(x) + x

        x = self.conv2(x) + x

        x = self.conv3(x) + x

        x = self.conv4(x) + x

        x = x.squeeze(0).permute(1, 2, 0)

        return x


class MyLayerNorm(nn.Module):
    def __init__(self):
        super().__init__()

        self.eps = 1e-4
    def forward(self, x):
        mean = torch.mean(x, dim=(0, 1, 2), keepdim=True)
        var = torch.var(x, dim=(0, 1, 2), keepdim=True)

        std = torch.sqrt(var + self.eps)

        #print(x.shape, mean.shape, std.shape)
        #exit()
        x =  (x - mean) / std

        return x

class DFOLDIpaScore(nn.Module):

    def __init__(self, model_conf, diffuser):
        super(DFOLDIpaScore, self).__init__()
        self._model_conf = model_conf
        ipa_conf = model_conf.ipa
        self._ipa_conf = ipa_conf
        self.diffuser = diffuser

        self.scale_pos = lambda x: x * ipa_conf.coordinate_scaling
        self.scale_rigids = lambda x: x.apply_trans_fn(self.scale_pos)

        self.unscale_pos = lambda x: x / ipa_conf.coordinate_scaling
        self.unscale_rigids = lambda x: x.apply_trans_fn(self.unscale_pos)

        self.trunk = nn.ModuleDict()

        for b in range(ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = InvariantPointAttention(ipa_conf)
            self.trunk[f'ln_{b}'] = MyLayerNorm()
            # module for rigids update and egde update
            self.trunk[f'bb_update_{b}'] = BackboneUpdate(ipa_conf.c_s*5)

        self.trunk[f'conv_0'] = ConvNet(ipa_conf.c_s*5)

        #self.trunk[f'temp_0'] = TempNet(ipa_conf.c_s*4)

        # module for angle prediction
        self.angle_resnet = AngleResnet(c_in=ipa_conf.c_s*5,c_hidden=ipa_conf.c_s*5,no_blocks=2,no_angles=7,epsilon=1e-12) 
        # self.atom_embeder = AtomEmbed(model_conf.node_embed_size)

        self.force_embeder = nn.Sequential(
            nn.Linear(3, model_conf.node_embed_size),
            nn.SiLU(),
            nn.Linear(model_conf.node_embed_size,model_conf.node_embed_size),
            #nn.LayerNorm((9, 40, model_conf.node_embed_size)),
            MyLayerNorm(),
            nn.SiLU()
        )
        self.vel_embeder = nn.Sequential(
            nn.Linear(3, model_conf.node_embed_size),
            nn.SiLU(),
            nn.Linear(model_conf.node_embed_size,model_conf.node_embed_size),
            #nn.LayerNorm((9, 40, model_conf.node_embed_size)),
            MyLayerNorm(),
            nn.SiLU()
        )
        self.index_embeder = nn.Sequential(
            nn.Linear(1, model_conf.node_embed_size),
            nn.SiLU(),
            nn.Linear(model_conf.node_embed_size,model_conf.node_embed_size),
            #nn.LayerNorm((1, 40, model_conf.node_embed_size)),
            MyLayerNorm(),
            nn.SiLU()
        )
        self.rigid_embeder = nn.Sequential(
            nn.Linear(7, model_conf.node_embed_size),
            nn.SiLU(),
            nn.Linear(model_conf.node_embed_size,model_conf.node_embed_size),
            #nn.LayerNorm((9, 40, model_conf.node_embed_size)),
            MyLayerNorm(),
            nn.SiLU()
        )

        self.angle_embeder = nn.Sequential(
            nn.Linear(14, model_conf.node_embed_size),
            nn.SiLU(),
            nn.Linear(model_conf.node_embed_size,model_conf.node_embed_size),
            MyLayerNorm(),
            nn.SiLU()
        )

    def forward(self, init_node_embed, edge_embed, input_feats,drop_ref=False):
        '''
        init_node_embed: [F,N,D] node features from embeder with diffuser t embeddings
        edge_embed: edge features from embeder without diffuser t embeddings, since we update egde features with node features
        input_feats: input infomation from dataloader
        '''
        # # position embedding add in the beginning
        # init_node_embed = self.init_time_position(init_node_embed.permute([1,0,2]))
        # init_node_embed = init_node_embed.permute([1,0,2])

        # initialize the input tensors
        diffuer_time_t = input_feats['t']
        node_mask = input_feats['res_mask'].type(torch.float32)
        diffuse_mask = (1 - input_feats['fixed_mask'].type(torch.float32)) * node_mask
        edge_mask = node_mask[..., None] * node_mask[..., None, :]
        init_frames = input_feats['rigids_t'].type(torch.float32)
        init_rigids = Rigid.from_tensor_7(init_frames)

        # create reference here
        nf = input_feats['rigids_0'].shape[0]

        curr_rigids = torch.cat([input_feats['rigids_0'][:-1], input_feats['rigids_0'][-2:-1]], dim=0) #use reference initialize current prediction

        force = input_feats['force'].to(input_feats['rigids_0'].dtype)
        force = torch.cat([force[:-1], force[-2:-1]], dim=0) #use reference initialize current prediction
        force_embed = self.force_embeder(force)

        vel = input_feats['vel'].to(input_feats['rigids_0'].dtype)
        vel = torch.cat([vel[:-1], vel[-2:-1]], dim=0) #use reference initialize current prediction
        vel_embed = self.vel_embeder(vel)

        node_embed = input_feats['seq_idx'][0:1].unsqueeze(-1)
        node_embed = node_embed.to(input_feats['node_repr'].dtype)
        node_embed = self.index_embeder(node_embed).expand(nf, -1, -1)

        node_embed = node_embed + input_feats['expand_node_repr'].clone()
        edge_embed = input_feats['expand_edge_repr'].clone()

        node_mask = input_feats['res_mask'].type(torch.float32)

  
        angle = input_feats['torsion_angles_sin_cos'].to(input_feats['rigids_0'].dtype)
        angle_mask = input_feats['torsion_angles_mask'].to(input_feats['rigids_0'].dtype)
        angle = (angle * angle_mask.unsqueeze(-1)).to(input_feats['rigids_0'].dtype)
        angle = torch.cat([angle[:-1], angle[-2:-1]], dim=0) #use reference initialize current prediction
        na = angle.shape[-2]
        angle = angle.reshape(nf, -1, na*2)
        angle_embed = self.angle_embeder(angle)

        for b in range(self._ipa_conf.num_blocks):

            spatial_curr_rigids = curr_rigids.clone()

            #spatial_curr_rigids_norm = spatial_curr_rigids - spatial_curr_rigids[0:1]
            #rigids_embed_norm = self.rigid_embeder_norm(spatial_curr_rigids_norm)
            #node_feat_norm = torch.cat([rigids_embed_norm, node_embed, force_embed, vel_embed], dim=-1)
            #node_feat_norm = self.trunk[f'temp_0'](node_feat_norm)

            rigids_embed = self.rigid_embeder(spatial_curr_rigids)
            all_ipa_embed = self.trunk[f'ipa_{b}'](node_embed, edge_embed, Rigid.from_tensor_7(spatial_curr_rigids), node_mask)
            all_ipa_embed = self.trunk[f'ln_{b}'](all_ipa_embed)
            node_feat = torch.cat([rigids_embed, all_ipa_embed, force_embed, vel_embed, angle_embed], dim=-1)
            #node_feat = torch.cat([all_ipa_embed, all_ipa_embed, all_ipa_embed], dim=-1)

            spatial_curr_rigids = Rigid.from_tensor_7(spatial_curr_rigids)
            node_feat = self.trunk[f'conv_0'](node_feat)

            #node_feat = node_feat + node_feat_norm

            rigid_update = self.trunk[f'bb_update_{b}'](node_feat) 

            rigid_update[:-1] = rigid_update[:-1] * 0.0 # don't update reference

            curr_rigids = Rigid.from_tensor_7(curr_rigids)
            curr_rigids = curr_rigids.compose_q_update_vec(rigid_update, diffuse_mask[..., None])
            curr_rigids = curr_rigids.to_tensor_7()

            if b == 0:
                init_node_feat = node_feat.clone()

        unorm_angles, angles = self.angle_resnet(node_feat, init_node_feat)

        curr_rigids = Rigid.from_tensor_7(curr_rigids)
        
        # # # 
        rot_score = self.diffuser.calc_rot_score(
            init_rigids.get_rots(),
            curr_rigids.get_rots(),
            diffuer_time_t
        )
        rot_score = rot_score * node_mask[..., None]

        curr_rigids = self.unscale_rigids(curr_rigids)
        trans_score = self.diffuser.calc_trans_score(
            init_rigids.get_trans(),
            curr_rigids.get_trans(),
            diffuer_time_t[:, None, None],
            use_torch=True,
        )
        trans_score = trans_score * node_mask[..., None]
        # merge the outputs 
        model_out = {
            'angles': angles,
            'unorm_angles':unorm_angles,
            'rot_score': rot_score,
            'trans_score': trans_score,
            'final_rigids': curr_rigids,
            'rigid_update':rigid_update
        }
        return model_out

