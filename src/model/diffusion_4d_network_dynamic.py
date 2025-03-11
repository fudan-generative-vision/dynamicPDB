"""Score network module."""
import torch
import math
from torch import nn
from torch.nn import functional as F
from openfold.utils import feats
from src.data import utils as du
from src.data import all_atom
from src.model import diffusion_4d_ipa_pytorch_dynamic
import functools as fn
from openfold.utils.tensor_utils import batched_gather
from typing import Dict, Text, Tuple

import torch
from src.model.utils import get_timestep_embedding
from openfold.np import residue_constants as rc
Tensor = torch.Tensor

class DFOLDv2_Embeder(nn.Module):

    def __init__(self, model_conf):
        super(DFOLDv2_Embeder, self).__init__()
        self._model_conf = model_conf
        self._embed_conf = model_conf.embed
        # Time step embedding
        diffuser_time_embed_size = self._embed_conf.index_embed_size
        node_embed_size = self._model_conf.node_embed_size
        edge_embed_size = self._model_conf.edge_embed_size
        time_embed_size = node_embed_size
        self.timestep_embed = fn.partial(
            get_timestep_embedding,
            embedding_dim=time_embed_size,
        )

        self.node_timestep_proj= nn.Sequential(
                    nn.Linear(time_embed_size,node_embed_size//2),
                    nn.SiLU(),
                    nn.Linear(node_embed_size//2,node_embed_size),
                )
        self.node_ln = nn.LayerNorm(node_embed_size)


        self.edge_timestep_proj = nn.Sequential(
                    nn.Linear(time_embed_size,edge_embed_size//2),
                    nn.SiLU(),
                    nn.Linear(edge_embed_size//2,edge_embed_size),
        )
        self.edge_ln = nn.LayerNorm(edge_embed_size)


    def forward(self,node_repr,edge_repr,seq_idx,t):
        """Embeds a 
        Args:
            node_repr: [B, N, D_node] node features from FOLD model like GeoForm(from OmegaFold)
            edge_repr: [B, N, N, D_edge] edge features from FOLD model like GeoForm(from OmegaFold)
            seq_idx: [..., N] Positional sequence index for each residue.
            t: Sampled t in [0, 1].

        Returns:
            node_embed: [B, N, D_node]
            edge_embed: [B, N, N, D_edge]
        """
        num_batch, num_res = seq_idx.shape
        t_embed = self.timestep_embed(t)
        # processing node
        node_t_step_embedings = self.node_timestep_proj(t_embed)
        node_t_step_embedings = torch.tile(node_t_step_embedings[:, None, :], (num_batch, num_res, 1)) # (B,Nr,32)

        node_embed = self.node_ln(node_repr)
        ref_node_repr = node_embed[0]#self.ln(node_repr[0])
        node_embed = node_embed+node_t_step_embedings

        # processing edge
        edge_embed = edge_repr.reshape(num_batch,num_res*num_res,-1)
        edge_t_step_embedings = self.edge_timestep_proj(t_embed)
        edge_t_step_embedings = torch.tile(edge_t_step_embedings[:, None, :], (num_batch, num_res*num_res, 1)) # (B,Nr,32)

        edge_embed = self.edge_ln(edge_embed)
        ref_edge_repr = edge_embed[0].reshape(num_res,num_res,-1)
        edge_embed = edge_embed+edge_t_step_embedings
        edge_embed = edge_embed.reshape(num_batch,num_res,num_res,-1)


 
        return node_embed, edge_embed ,ref_node_repr,ref_edge_repr,t_embed

class FullScoreNetwork(nn.Module):

    def __init__(self, model_conf, diffuser):
        super(FullScoreNetwork, self).__init__()
        self._model_conf = model_conf
        self.embedding_layer = DFOLDv2_Embeder(model_conf)

        self.diffuser = diffuser
        self.score_model = diffusion_4d_ipa_pytorch_dynamic.DFOLDIpaScore(model_conf, diffuser)
        self.expand_node = nn.Linear(256, model_conf.node_embed_size)
        self.expand_edge = nn.Linear(128, model_conf.edge_embed_size)
        
    def _apply_mask(self, aatype_diff, aatype_0, diff_mask):
        return diff_mask * aatype_diff + (1 - diff_mask) * aatype_0

    def forward(self, input_feats,drop_ref=False,is_training=True):
        """Forward computes the reverse diffusion conditionals p(X^t|X^{t+1})
        for each item in the batch

        Args:
            X: the noised samples from the noising process, of shape [Batch, N, D].
                Where the T time steps are t=1,...,T (i.e. not including the un-noised X^0)

        Returns:
            model_out: dictionary of model outputs.
        """
        # total_frame = self._model_conf.frame_time+self._model_conf.ref_number+self._model_conf.motion_number
        # Frames as [batch, res, 7] tensors.
        # bb_mask = input_feats['res_mask'].type(torch.float32) #.unsqueeze(0).expand(self._model_conf.frame_time, -1).type(torch.float32)  # [B, N]
        fixed_mask = input_feats['fixed_mask'].type(torch.float32) #.unsqueeze(0).expand(self._model_conf.frame_time, -1).type(torch.float32)
        # edge_mask = bb_mask[..., None] * bb_mask[..., None, :]

        num_res = input_feats['node_repr'].shape[0]
        input_feats['expand_node_repr'] = self.expand_node(input_feats['node_repr'])
        input_feats['expand_edge_repr'] = self.expand_edge(input_feats['edge_repr'].reshape(num_res*num_res, -1)).reshape(num_res, num_res, -1)

        # Initial embeddings of positonal and relative indices.
        if self._model_conf.embed.DFOLDv2_embedder:
            init_node_embed, init_edge_embed,ref_node_repr,ref_edge_repr,t_embed = self.embedding_layer(
                node_repr=input_feats['expand_node_repr'].unsqueeze(0).expand(self._model_conf.frame_time, -1,-1),#
                edge_repr=input_feats['expand_edge_repr'].unsqueeze(0).expand(self._model_conf.frame_time, -1,-1,-1),#
                seq_idx=input_feats['seq_idx'],#.unsqueeze(0).expand(self._model_conf.frame_time, -1),
                t=input_feats['t'],
            )
            input_feats.update({'ref_node_repr':ref_node_repr,"ref_edge_repr":ref_edge_repr,'t_embed':t_embed})
        else:
            pass
            # maybe other embedder
            
        # edge_embed = init_edge_embed * edge_mask[..., None]# all the same
        # node_embed = init_node_embed * node_mask[..., None] # all the same
        # Run main network
        model_out = self.score_model(init_node_embed, init_edge_embed, input_feats,drop_ref=drop_ref,is_training=is_training) 
        gt_angles = input_feats['torsion_angles_sin_cos'] #in SE3 [..., 2, :] # only angle psi is used，here use the all 7 angles
        angles_pred = self._apply_mask(model_out['angles'], gt_angles, 1 - fixed_mask[..., None, None] ) # could del since fixed_masks always equal to  1
        unorm_angles = self._apply_mask(model_out['unorm_angles'], gt_angles, 1 - fixed_mask[..., None, None]) # could del since fixed_masks always equal to  1
        pred_out = {
            'angles': angles_pred,
            'unorm_angles': unorm_angles,
            'rot_score': model_out['rot_score'],
            'trans_score': model_out['trans_score'],
        }
        rigids_pred = model_out['final_rigids']


        
        pred_out['rigids'] = rigids_pred.to_tensor_7()
        # convert rigids and angles to frames
        all_frames = feats.torsion_angles_to_frames(rigids_pred,angles_pred,  
                                                    input_feats['aatype'],
                                                    all_atom.DEFAULT_FRAMES.to(angles_pred.device))
        # convert frame to atom14 with idealize amino acid 
        atom14_pos = all_atom.frames_to_atom14_pos(all_frames,input_feats['aatype']) 
        # change atom14 to atom37
        atom37_pos,atom37_mask = atom14_to_atom37(atom14_pos, input_feats['aatype'] ) 

        # bb_representations = all_atom.compute_backbone(rigids_pred, psi_pred)
        pred_out['atom37'] = atom37_pos.to(rigids_pred.device)
        pred_out['atom14'] = atom14_pos.to(rigids_pred.device)
        pred_out['rigid_update']= model_out['rigid_update']
        # angles torch.Size([1, 38, 7, 2])
        # unorm_angles torch.Size([1, 38, 7, 2])
        # rot_score torch.Size([1, 38, 3])
        # trans_score torch.Size([1, 38, 3])
        # rigids torch.Size([1, 38, 7])
        # atom37 torch.Size([1, 38, 37, 3])
        # atom14 torch.Size([1, 38, 14, 3])
        # rigid_update torch.Size([1, 38, 6])
        return pred_out


def get_rc_tensor(rc_np, aatype):
    return torch.tensor(rc_np, device=aatype.device)[aatype]

def atom14_to_atom37(
    atom14_data: torch.Tensor,  # (*, N, 14, ...)
    aatype: torch.Tensor # (*, N)
) -> Tuple:    # (*, N, 37, ...)
    """Convert atom14 to atom37 representation."""
    idx_atom37_to_atom14 = get_rc_tensor(rc.RESTYPE_ATOM37_TO_ATOM14, aatype).long()
    no_batch_dims = len(aatype.shape) - 1
    atom37_data = batched_gather(
        atom14_data, 
        idx_atom37_to_atom14, 
        dim=no_batch_dims + 1, 
        no_batch_dims=no_batch_dims + 1
    )
    atom37_mask = get_rc_tensor(rc.RESTYPE_ATOM37_MASK, aatype) 
    if len(atom14_data.shape) == no_batch_dims + 2:
        atom37_data *= atom37_mask
    elif len(atom14_data.shape) == no_batch_dims + 3:
        atom37_data *= atom37_mask[..., None].to(dtype=atom37_data.dtype)
    else:
        raise ValueError("Incorrectly shaped data")
    return atom37_data, atom37_mask
