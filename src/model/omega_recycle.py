import sys
sys.path.append('/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/zsy_43187/protein/workspace/chengkaihui/code/D-FOLD/')

import torch
from torch import nn
import typing
from omegafold.decode import TorsionAngleHead,StructureCycle
from omegafold.geoformer import GeoFormer
from omegafold.embedders import RecycleEmbedder
import argparse
import typing
from omegafold import modules, utils
from openfold.utils.rigid_utils import Rigid,Rotation
from openfold.np.residue_constants  import  restype_atom14_mask
from contextlib import ExitStack

class StructureModule(modules.OFModule):
    """Jumper et al. (2021) Suppl. Alg. 20 'StructureModule'"""

    def __init__(self, cfg: argparse.Namespace):
        super(StructureModule, self).__init__(cfg)
        self.node_norm = nn.LayerNorm(cfg.node_dim)
        self.edge_norm = nn.LayerNorm(cfg.edge_dim)
        self.init_proj = nn.Linear(cfg.node_dim, cfg.node_dim)

        self.cycles = nn.ModuleList(
            [StructureCycle(cfg) for _ in range(cfg.num_cycle)]
        )
        self.torsion_angle_pred = TorsionAngleHead(cfg)

    def forward(
            self,
            node_repr: torch.Tensor, edge_repr: torch.Tensor,
            mask: torch.Tensor,fasta: torch.Tensor,init_frames
    ):
        """
        Jumper et al. (2021) Suppl. Alg. 20 "StructureModule"

        Args:
            node_repr: node representation tensor of shape [num_res, dim_node]
            edge_repr: edge representation tensor of shape [num_res, dim_edge]
            fasta: the tokenized sequence of the input protein sequence
            mask

        Returns:
            node_repr: The current node representation tensor for confidence
                of shape [num_res, dim_node]
            dictionary containing:
                final_atom_positions: the final atom14 positions,
                    of shape [num_res, 14, 3]
                final_atom_mask: the final atom14 mask,
                    of shape [num_res, 14]

        """
        node_repr = self.node_norm(node_repr)
        edge_repr = self.edge_norm(edge_repr)

        init_node_repr = node_repr
        node_repr = self.init_proj(node_repr)
        # Initialize the initial frames with Black-hole Jumper et al. (2021)
        backbone_frames = utils.AAFrame.default_init(
            *node_repr.shape[:-1],
            unit='nano',
            device=self.device,
            mask=mask.bool()
        )
        # print('hello Black-hole')
        # backbone_frames = utils.AAFrame.from_tensor(init_frames,unit='nano')
            # translation = init_rigids._trans,
            # rotation=init_rigids._rots.get_rot_mats(),
            # unit='nano',
            # mask=mask.bool()
        # )
        #     print(backbone_frames.translation[0,:2],init_rigids._trans[0,:2])
        #     print(backbone_frames.rotation[0,:2],init_rigids._rots.get_rot_mats()[0,:2])
        # exit()

        # print((init_rigids._rots.get_rot_mats()).shape,init_rigids._trans.shape,backbone_frames.rotation.shape,backbone_frames.translation.shape,'\n','='*10)
        # exit()

        for layer in self.cycles:
            node_repr, backbone_frames = layer(
                node_repr, edge_repr, backbone_frames
            )

        torsion_angles_sin_cos = self.torsion_angle_pred(
            representations_list=[node_repr, init_node_repr],
        )

        torsion_angles_mask = torch.ones_like(
            torsion_angles_sin_cos[..., 0], dtype=torch.bool
        )
        backbone_frames = backbone_frames.to_angstrom(in_place=False)

        frames8 = backbone_frames.expand_w_torsion(
            torsion_angles=torsion_angles_sin_cos,
            torsion_angles_mask=torsion_angles_mask,
            fasta=fasta
        )
        # print(frames8.shape)
        # exit()
        pos14, mask14 = frames8.expanded_to_pos(fasta)
        # convert to openfold form
        rots = Rotation(rot_mats=backbone_frames.rotation)
        # if rots.device == torch.device('cuda:0'):
        #     print(backbone_frames.rotation,'\n',backbone_frames.translation)
        return node_repr,{
            "final_frames": Rigid(rots=rots,trans=backbone_frames.translation).to_tensor_7(),
            "final_atom_positions": pos14,
            # "final_atom_mask": mask14,
            "angles":torsion_angles_sin_cos
        }




if __name__ == '__main__':
    def init_structure_module(model,weight_path='/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/zsy_43187/protein/workspace/chengkaihui/code/D-FOLD/ckh_tool/release1.pt'):
        weight_dict = torch.load(weight_path,map_location=torch.device('cpu'))
        weight_keys = weight_dict.keys()

        filtered_state_dict = {}
        for k in model.state_dict().keys():
            key_in_pretrain_model = 'omega_fold_cycle.structure_module.'+k 
            if key_in_pretrain_model in weight_keys:
                filtered_state_dict.update({k:weight_dict[key_in_pretrain_model]})
        
        # Load the filtered state_dict into your model
        model.load_state_dict(filtered_state_dict)
        return model
    

    # def create_initial_prev_dict(num_res: int) -> typing.Dict[str, torch.Tensor]:
    #     return {
    #         "prev_node": torch.zeros(
    #             [num_res, self.cfg.node_dim],
    #             device=self.device, dtype=torch.float
    #         ),
    #         "prev_edge": torch.zeros(
    #             [num_res, num_res, self.cfg.edge_dim],
    #             device=self.device, dtype=torch.float
    #         ),
    #         "prev_x": torch.zeros(
    #             [num_res, 14, 3],
    #             device=self.device, dtype=torch.float
    #         ),
    #         "prev_frames": utils.AAFrame.default_init(
    #             num_res, 8, unit="Angstrom", device=self.device
    #         )
    # }

    from omegafold import config,pipeline
    args, forward_config = pipeline.get_args()
    cfg = config.make_config()


    recycle_embedder = RecycleEmbedder(cfg).cuda()
    struct_module = StructureModule(cfg.struct).cuda()
    geoformer = GeoFormer(cfg).cuda()

    node_repr=torch.rand([1,105,256]).cuda()
    edge_repr=torch.rand([105, 105, 128]).cuda()
    fasta=torch.randint(0,20,size=[105]).cuda()
    mask=torch.ones(size=[1,105]).cuda()

    # node_repr=torch.rand([1,105,256]).cuda()
    # edge_repr=torch.rand([105, 105, 128]).cuda()
    # fasta=torch.randint(0,20,size=[105]).cuda()
    # mask=torch.ones(size=[1,105]).cuda()

    recycle_num = 4
    
    prev_node = torch.zeros([105,256]).cuda()
    prev_edge = torch.zeros([105, 105, 128]).cuda()
    prev_x = torch.zeros([105, 14, 3]).cuda()

    for recycle_idx in range(recycle_num):
        with ExitStack() if recycle_idx == recycle_num - 1 else torch.no_grad():
            # print(prev_edge.shape, prev_node.shape)
            node_repr, edge_repr = recycle_embedder(
                    fasta=fasta,
                    prev_node= prev_node  ,#prev_dict.pop('prev_node'), # [num_res, node_repr_dim]
                    prev_edge= prev_edge , #prev_dict.pop('prev_edge'), # [num_res, num_res, edge_repr_dim]
                    prev_x= prev_x, #prev_dict.pop('prev_x'),  # [num_res,14, 3]  pseudo beta coordinates
                    node_repr=node_repr,
                    edge_repr=edge_repr
                )
            print(edge_repr.shape, node_repr.shape)
            node_repr, edge_repr, final_node = geoformer(node_repr=node_repr,
                    edge_repr=edge_repr,
                    mask=mask,
                    fwd_cfg=forward_config)
            
            print(edge_repr.shape, node_repr.shape)
            exit()
            # structure model
            # position 14
            # structure mdoel ended
            prev_node = node_repr[..., 0, :, :]
            prev_edge = edge_repr
            prev_x = torch.rand([105, 14, 3]).cuda() # position 14
            print(edge_repr.shape, final_node.shape)



    

    exit()


    
    
    model = init_structure_module(model)
    # torch.Size([4, 105, 256]) torch.Size([4, 105, 105, 128])
    # torch.Size([4, 105]) torch.Size([4, 105])
    print(model)

    node_repr,x = model(
        node_repr=torch.rand([4,105,384]), 
        edge_repr=torch.rand([4,105, 105, 128]),
        fasta=torch.randint(0,20,size=[4,105]),
        mask=torch.ones(size=[4,105])

    )
    print(node_repr.shape)
    for k in x.keys():
        print(k,x[k].shape)

    exit()
