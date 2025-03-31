"""Score network module."""
import torch
import math
from torch import nn
from torch.nn import functional as F
from openfold.utils import feats
from src.data import utils as du
from src.data import all_atom
from src.model import ipa_pytorch_dynamic
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

        #print(node_embed.shape, node_t_step_embedings.shape)
        #exit()
        #node_embed = node_embed+node_t_step_embedings

        # processing edge
        edge_embed = edge_repr.reshape(num_batch,num_res*num_res,-1)
        edge_t_step_embedings = self.edge_timestep_proj(t_embed)
        edge_t_step_embedings = torch.tile(edge_t_step_embedings[:, None, :], (num_batch, num_res*num_res, 1)) # (B,Nr,32)

        edge_embed = self.edge_ln(edge_embed)
        ref_edge_repr = edge_embed[0].reshape(num_res,num_res,-1)
        #edge_embed = edge_embed+edge_t_step_embedings
        edge_embed = edge_embed.reshape(num_batch,num_res,num_res,-1)


 
        return node_embed, edge_embed ,ref_node_repr,ref_edge_repr,t_embed


class DFOLDv2_Embederv2(nn.Module):

    def __init__(self, model_conf):
        #print('hello DFOLDv2_Embederv2')
        super(DFOLDv2_Embederv2, self).__init__()
        self._model_conf = model_conf
        self._embed_conf = model_conf.embed
        # Time step embedding
        diffuser_time_embed_size = self._embed_conf.index_embed_size
        node_embed_size = self._model_conf.node_embed_size
        edge_embed_size = self._model_conf.edge_embed_size

        self.timestep_proj = fn.partial(
            get_timestep_embedding,
            embedding_dim=self._embed_conf.index_embed_size,
        )

        self.node_timestep_embedder= nn.Sequential(
                    nn.Linear(self._embed_conf.index_embed_size,node_embed_size//2),
                    nn.SiLU(),
                    nn.Linear(node_embed_size//2,node_embed_size),
                )
        self.edge_timestep_embedder= nn.Sequential(
            nn.Linear(self._embed_conf.index_embed_size,edge_embed_size//2),
            nn.SiLU(),
            nn.Linear(edge_embed_size//2,edge_embed_size),
        )
        
        self.node_ln = nn.LayerNorm(node_embed_size)
        self.edge_ln = nn.LayerNorm(edge_embed_size)

        # amino acid sequence embedder
        self.node_block_out_channels = [node_embed_size,512,256]
        self.edge_block_out_channels = [edge_embed_size,256,128]

        self.node_blocks = nn.ModuleList([])
        self.edge_blocks = nn.ModuleList([])

        for i in range(len(self.node_block_out_channels)-1):
            node_channel_in,node_channel_out = self.node_block_out_channels[i],self.node_block_out_channels[i+1]
            edge_channel_in,edge_channel_out = self.edge_block_out_channels[i],self.edge_block_out_channels[i+1]

            self.node_blocks.append(
                nn.Conv1d(in_channels=node_channel_in,out_channels=node_channel_out,kernel_size=3,padding=1)
                )
            self.edge_blocks.append(
                nn.Conv2d(in_channels=edge_channel_in,out_channels=edge_channel_out,kernel_size=3,padding=1)
            )
        
        self.node_out = zero_module(
            nn.Conv1d(in_channels=self.node_block_out_channels[-1],out_channels=node_embed_size,kernel_size=3,padding=1)
        )

        self.edge_out = zero_module(
            nn.Conv2d(in_channels=self.edge_block_out_channels[-1],out_channels=edge_embed_size,kernel_size=3,padding=1)
        )

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
        proj_t = self.timestep_proj(t)

        node_t_step_embedings = self.node_timestep_embedder(proj_t)
        node_t_step_embedings = torch.tile(node_t_step_embedings[:, None, :], (num_batch, num_res, 1)) # (B,Nr,32)

        edge_t_step_embedings = self.edge_timestep_embedder(proj_t)
        edge_t_step_embedings = torch.tile(edge_t_step_embedings[:, None,None, :], (num_batch, num_res, num_res, 1)) # (B,Nr,32)

        node_embed_init = self.node_ln(node_repr+node_t_step_embedings)
        edge_embed_init = self.edge_ln(edge_repr+edge_t_step_embedings)
        
        node_embed_init = node_embed_init.permute([0,2,1])
        edge_embed_init = edge_embed_init.permute([0,3,1,2])
        for i in range(len(self.node_block_out_channels)-1):
            if i==0:
                node_embed = self.node_blocks[i](node_embed_init)
                edge_embed = self.edge_blocks[i](edge_embed_init)
            else:
                node_embed = self.node_blocks[i](node_embed)
                edge_embed = self.edge_blocks[i](edge_embed)

            node_embed = F.silu(node_embed)
            edge_embed = F.silu(edge_embed)

        node_embed = self.node_out(node_embed)
        edge_embed = self.edge_out(edge_embed)

        node_embed = node_embed+node_embed_init
        edge_embed = edge_embed+edge_embed_init

        # ref_node_repr =  node_repr[0]#self.ln(node_repr[0])
        node_embed = node_embed.permute([0,2,1])
        edge_embed = edge_embed.permute([0,2,3,1])
        return node_embed, edge_embed #,ref_node_repr


class Embedder(nn.Module):

    def __init__(self, model_conf):
        super(Embedder, self).__init__()
        self._model_conf = model_conf
        self._embed_conf = model_conf.embed

        # Time step embedding
        index_embed_size = self._embed_conf.index_embed_size
        t_embed_size = index_embed_size
        node_embed_dims = t_embed_size + 1
        edge_in = (t_embed_size + 1) * 2

        # Sequence index embedding
        node_embed_dims += index_embed_size
        # node_embed_dims += self._embed_conf.node_repr_dim
        edge_in += index_embed_size
        # edge_in += self._embed_conf.edge_repr_dim

        node_embed_size = self._model_conf.node_embed_size
        self.node_embedder = nn.Sequential(
            nn.Linear(node_embed_dims, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.LayerNorm(node_embed_size),
        )

        if self._embed_conf.embed_self_conditioning:
            edge_in += self._embed_conf.num_bins
        edge_embed_size = self._model_conf.edge_embed_size
        self.edge_embedder = nn.Sequential(
            nn.Linear(edge_in, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.LayerNorm(edge_embed_size),
        )
        if not self._model_conf.embed.skip_feature:
            self.node_repr_norm = nn.LayerNorm(node_embed_size)
            self.edge_repr_norm = nn.LayerNorm(edge_embed_size)

        if self._model_conf.dynamics:

            if self._model_conf.ipa.temporal:
                self.frame_timestep_embedder_first=fn.partial(
                    get_frame_embedding,
                    embed_size=node_embed_size//2
                )
                self.frame_timestep_embedder= nn.Sequential(
                    nn.Linear(node_embed_size//2,node_embed_size//2),
                    nn.SiLU(),
                    nn.Linear(node_embed_size//2,node_embed_size),
                    nn.LayerNorm(node_embed_size)
                )

        self.timestep_embedder = fn.partial(
            get_timestep_embedding,
            embedding_dim=self._embed_conf.index_embed_size,

        )
        self.index_embedder = fn.partial(
            get_index_embedding,
            embed_size=self._embed_conf.index_embed_size
        )

    def _cross_concat(self, feats_1d, num_batch, num_res):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res**2, -1])

    def forward(
            self,
            node_repr,
            edge_repr,
            seq_idx,
            t,
            fixed_mask,
            self_conditioning_ca,
        ):
        """Embeds a set of inputs

        Args:
            seq_idx: [..., N] Positional sequence index for each residue.
            t: Sampled t in [0, 1].
            fixed_mask: mask of fixed (motif) residues.
            self_conditioning_ca: [..., N, 3] Ca positions of self-conditioning
                input.

        Returns:
            node_embed: [B, N, D_node]
            edge_embed: [B, N, N, D_edge]
        """
        num_batch, num_res = seq_idx.shape
        node_feats = []

        # Set time step to epsilon=1e-5 for fixed residues.
        fixed_mask = fixed_mask[..., None] # (B,Nr,1)
        if self._model_conf.dynamics:
            # frame_t = torch.arange(num_batch).to(node_repr.device)
            if self._model_conf.ipa.temporal:
                frame_t = torch.arange(num_batch).to(seq_idx.device)
                frame_t_embed = torch.tile(self.frame_timestep_embedder_first(frame_t)[:, None, :], (1, num_res, 1)) 
                
                frame_t_embed = self.frame_timestep_embedder(frame_t_embed)

            # print(frame_t_embed.shape)
            # print(self.frame_timestep_embedder)
            # print(self.frame_timestep_embedder(frame_t_embed).shape)
            # exit()
            # frame_t_embed = torch.tile(self.frame_timestep_embedder(frame_t)[:, None, :], (1, num_res, 1)) 

            t_embed = torch.tile(self.timestep_embedder(t)[:, None, :], (num_batch, num_res, 1)) # (B,Nr,32)
        else:
            t_embed = torch.tile(self.timestep_embedder(t)[:, None, :], (1, num_res, 1)) # (B,Nr,32)
        prot_t_embed = torch.cat([t_embed, fixed_mask], dim=-1) # (B,Nr,32+1)
        node_feats = [prot_t_embed]
        pair_feats = [self._cross_concat(prot_t_embed, num_batch, num_res)] # 把node的特征拼接起来 # (B,Nr**2,2(32+1)) 
        # Positional index features.
        node_feats.append(self.index_embedder(seq_idx)) # (B,Nr,32)
        # node_feats.append(node_repr) #append node repr

        rel_seq_offset = seq_idx[:, :, None] - seq_idx[:, None, :]
        rel_seq_offset = rel_seq_offset.reshape([num_batch, num_res**2])  # (B,Nr**2,32)

        pair_feats.append(self.index_embedder(rel_seq_offset)) # 根据相对位置计算position embedding

        # Self-conditioning distogram.
        if self._embed_conf.embed_self_conditioning:
            sc_dgram = du.calc_distogram(
                self_conditioning_ca,
                self._embed_conf.min_bin,
                self._embed_conf.max_bin,
                self._embed_conf.num_bins,
            )
            pair_feats.append(sc_dgram.reshape([num_batch, num_res**2, -1]))
        # pair_feats.append(edge_repr.reshape([num_batch, num_res**2, -1])) #append edge repr
        # node_feats.append(node_repr)
        # pair_feats.append(edge_repr.reshape([num_batch, num_res**2, -1]))
        node_embed = self.node_embedder(torch.cat(node_feats, dim=-1).float())# torch.Size([21, 153, 65])
        edge_embed = self.edge_embedder(torch.cat(pair_feats, dim=-1).float())# torch.Size([21, 23409, 120])
        edge_embed = edge_embed.reshape([num_batch, num_res, num_res, -1])
        # TODO here we add the feature from pretrained FOLD model
        # print(node_embed.shape,edge_embed.shape)
        # print(node_embed[0,2,:5],'\n',self.node_repr_norm(node_repr)[0,2,:5])
        # print(self.node_repr_norm.weight)
        # exit()
        if not self._model_conf.embed.skip_feature:
            node_embed += self.node_repr_norm(node_repr)
            normed_edge = edge_repr.reshape([num_batch, num_res**2, -1])
            # print(edge_embed[0,2,:10],'\n', self.edge_repr_norm(normed_edge).reshape([num_batch, num_res, num_res, -1])[0,2,:10])
            edge_embed += self.edge_repr_norm(normed_edge).reshape([num_batch, num_res, num_res, -1])
        # exit()
        # print(node_embed[0,2,:5],'\n',node_repr[0,2,:5])
        # f =nn.LayerNorm(256).cuda()
        # print(f(node_embed)[0,2,:5],'\n',f(node_repr)[0,2,:5])
        
        # exit()
        if self._model_conf.dynamics:
            if self._model_conf.ipa.temporal:
                node_embed=node_embed+frame_t_embed
            else:
                node_embed=node_embed
            # print('hello')
        return node_embed, edge_embed   


class ScoreNetwork(nn.Module):

    def __init__(self, model_conf, diffuser):
        super(ScoreNetwork, self).__init__()
        self._model_conf = model_conf

        self.embedding_layer = Embedder(model_conf)
        self.diffuser = diffuser
        self.score_model = ipa_pytorch_dynamic.IpaScore(model_conf, diffuser)

    def _apply_mask(self, aatype_diff, aatype_0, diff_mask):
        return diff_mask * aatype_diff + (1 - diff_mask) * aatype_0

    def forward(self, input_feats):
        """Forward computes the reverse diffusion conditionals p(X^t|X^{t+1})
        for each item in the batch

        Args:
            X: the noised samples from the noising process, of shape [Batch, N, D].
                Where the T time steps are t=1,...,T (i.e. not including the un-noised X^0)

        Returns:
            model_out: dictionary of model outputs.
        """
        # Frames as [batch, res, 7] tensors.
        bb_mask = input_feats['res_mask'].type(torch.float32)  # [B, N]
        fixed_mask = input_feats['fixed_mask'].type(torch.float32)
        edge_mask = bb_mask[..., None] * bb_mask[..., None, :]
        # Initial embeddings of positonal and relative indices.
        init_node_embed, init_edge_embed = self.embedding_layer(
            node_repr=input_feats['node_repr'],
            edge_repr=input_feats['edge_repr'],
            seq_idx=input_feats['seq_idx'],
            t=input_feats['t'],
            fixed_mask=fixed_mask,
            self_conditioning_ca=input_feats['sc_ca_t'],
        )
        edge_embed = init_edge_embed * edge_mask[..., None]
        node_embed = init_node_embed * bb_mask[..., None]

        # Run main network
        model_out = self.score_model(node_embed, edge_embed, input_feats)

        # Psi angle prediction
        gt_psi = input_feats['torsion_angles_sin_cos'][..., 2, :] # 只使用了psi其他的忽略了，
        psi_pred = self._apply_mask(
            model_out['psi'], gt_psi, 1 - fixed_mask[..., None])

        pred_out = {
            'psi': psi_pred,
            'rot_score': model_out['rot_score'],
            'trans_score': model_out['trans_score'],
        }
        rigids_pred = model_out['final_rigids']
        pred_out['rigids'] = rigids_pred.to_tensor_7()
        bb_representations = all_atom.compute_backbone(rigids_pred, psi_pred)
        pred_out['atom37'] = bb_representations[0].to(rigids_pred.device)
        pred_out['atom14'] = bb_representations[-1].to(rigids_pred.device)
        return pred_out



class FullScoreNetwork(nn.Module):

    def __init__(self, model_conf, diffuser):
        super(FullScoreNetwork, self).__init__()
        self._model_conf = model_conf
        # if self._model_conf.embed.DFOLDv2_embedder:
        self.embedding_layer = DFOLDv2_Embeder(model_conf)
        # elif self._model_conf.embed.DFOLDv2_embedderv2:
        #     self.embedding_layer = DFOLDv2_Embederv2(model_conf)
        # elif self._model_conf.embed.Embedderv3:
        #     self.embedding_layer = Embedderv3(model_conf)
        # else:
        #     self.embedding_layer = Embedder(model_conf)
        self.diffuser = diffuser
        self.score_model = ipa_pytorch_dynamic.DFOLDIpaScore(model_conf, diffuser)
        self.expand_node = nn.Linear(256, model_conf.node_embed_size)
        self.expand_edge = nn.Linear(128, model_conf.edge_embed_size)
        
    def _apply_mask(self, aatype_diff, aatype_0, diff_mask):
        return diff_mask * aatype_diff + (1 - diff_mask) * aatype_0

    def forward(self, input_feats,drop_ref=False):
        # info = ''
        # for k in input_feats.keys():
        #     info+=f' {k}:{input_feats[k].shape}:{type(input_feats[k])} ||'
        # print(info)
        # exit()
        """Forward computes the reverse diffusion conditionals p(X^t|X^{t+1})
        for each item in the batch

        Args:
            X: the noised samples from the noising process, of shape [Batch, N, D].
                Where the T time steps are t=1,...,T (i.e. not including the un-noised X^0)

        Returns:
            model_out: dictionary of model outputs.
        """
        # Frames as [batch, res, 7] tensors.
        bb_mask = input_feats['res_mask'].type(torch.float32) #.unsqueeze(0).expand(self._model_conf.frame_time, -1).type(torch.float32)  # [B, N]
        fixed_mask = input_feats['fixed_mask'].type(torch.float32) #.unsqueeze(0).expand(self._model_conf.frame_time, -1).type(torch.float32)
        edge_mask = bb_mask[..., None] * bb_mask[..., None, :]

        num_res = input_feats['node_repr'].shape[0]

        input_feats['expand_node_repr'] = self.expand_node(input_feats['node_repr'])
        input_feats['expand_edge_repr'] = self.expand_edge(input_feats['edge_repr'].reshape(num_res*num_res, -1)).reshape(num_res, num_res, -1)

        # Initial embeddings of positonal and relative indices.
        if self._model_conf.embed.DFOLDv2_embedder:
            init_node_embed, init_edge_embed,ref_node_repr,ref_edge_repr,t_embed = self.embedding_layer(
                node_repr=input_feats['expand_node_repr'].unsqueeze(0).expand(self._model_conf.frame_time, -1,-1),#
                edge_repr=input_feats['expand_edge_repr'].unsqueeze(0).expand(self._model_conf.frame_time, -1,-1,-1),#
                seq_idx=input_feats['seq_idx'],#.unsqueeze(0).expand(self._model_conf.frame_time, -1)
                t=input_feats['t'],
            )
            input_feats.update({'ref_node_repr':ref_node_repr,"ref_edge_repr":ref_edge_repr,'t_embed':t_embed})


        # elif self._model_conf.embed.DFOLDv2_embedderv2:
        #     init_node_embed, init_edge_embed = self.embedding_layer(
        #         node_repr=input_feats['node_repr'].unsqueeze(0).expand(self._model_conf.frame_time, -1,-1),#
        #         edge_repr=input_feats['edge_repr'].unsqueeze(0).expand(self._model_conf.frame_time, -1,-1,-1),#
        #         seq_idx=input_feats['seq_idx'],#.unsqueeze(0).expand(self._model_conf.frame_time, -1)
        #         t=input_feats['t'],
        #     )
        # elif self._model_conf.embed.Embedderv3:
        #     init_node_embed, init_edge_embed = self.embedding_layer(
        #         seq_idx=input_feats['seq_idx'],#.unsqueeze(0).expand(self._model_conf.frame_time, -1)
        #         t=input_feats['t'],
        #         fixed_mask=fixed_mask,
        #         self_conditioning_ca=input_feats['sc_ca_t'],
        #     )

        else:
            init_node_embed, init_edge_embed = self.embedding_layer(
                node_repr=input_feats['expand_node_repr'].unsqueeze(0).expand(self._model_conf.frame_time, -1,-1),#
                edge_repr=input_feats['expand_edge_repr'].unsqueeze(0).expand(self._model_conf.frame_time, -1,-1,-1),#
                seq_idx=input_feats['seq_idx'],#.unsqueeze(0).expand(self._model_conf.frame_time, -1)
                t=input_feats['t'],
                fixed_mask=fixed_mask,
                self_conditioning_ca=input_feats['sc_ca_t'],
            )
            
        edge_embed = init_edge_embed * edge_mask[..., None]# all the same
        node_embed = init_node_embed * bb_mask[..., None] # all the same
        # Run main network
        model_out = self.score_model(node_embed, edge_embed, input_feats,drop_ref=drop_ref) 

        gt_angles = input_feats['torsion_angles_sin_cos'] #in SE3 [..., 2, :] # only angle psi is used，here use the all 7 angles
        angles_pred = self._apply_mask(model_out['angles'], gt_angles, 1 - fixed_mask[..., None, None]) # could del since fixed_masks always equal to  1
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
        atom37_pos,atom37_mask = atom14_to_atom37(atom14_pos, input_feats['aatype']) 
        # atom37_pos,atom37_mask = atom14_to_atom37(model_out['final_atom_positions'], input_feats['aatype'])  # omega structure

        # bb_representations = all_atom.compute_backbone(rigids_pred, psi_pred)
        pred_out['atom37'] = atom37_pos.to(rigids_pred.device)
        pred_out['atom14'] = atom14_pos.to(rigids_pred.device)
        pred_out['rigid_update']= model_out['rigid_update']

        return pred_out


    def debug_foward(self,input_feats,drop_ref=False):
        used_params = set()
        
        def hook(module, input, output):
            for param in module.parameters():
                used_params.add(param)

        # 注册 hook
        hooks = []
        for module in self.modules():
            if isinstance(module, nn.Module):
                hooks.append(module.register_forward_hook(hook))

        # 前向传播
        output = self.forward(input_feats,drop_ref)

        # 清除 hook
        for h in hooks:
            h.remove()

        return output, used_params

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
