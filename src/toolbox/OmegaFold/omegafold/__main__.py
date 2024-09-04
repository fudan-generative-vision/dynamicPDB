# -*- coding: utf-8 -*-
# =============================================================================
# Copyright 2022 HeliXon Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""
The main function to run the prediction
"""
# =============================================================================
# Imports
# =============================================================================
import gc
import logging
import os
import sys
import time
from numpy import save

import torch

import omegafold as of
from . import pipeline
import _pickle as cPickle


# =============================================================================
# Functions
# =============================================================================

class OmegaFoldModel:
    @torch.no_grad()
    def __init__(self, model_weights_path, device="cuda"):
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        self.args, self.forward_config = pipeline.get_args()
        self.args.device = device
        # get the model
        logging.info(f"Constructing OmegaFold")
        self.model = of.OmegaFold(of.make_config())
        # print() #  'omega_fold_cycle.confidence_head.network.4.bias'
        # print('===='*100)

        state_dict = torch.load(model_weights_path, map_location='cpu')
        if state_dict is None:
            logging.warning("Inferencing without loading weight")
        else:
            if "model" in state_dict:
                state_dict = state_dict.pop("model")
            # weight_dict = {}
            # for k in self.model.state_dict().keys():
            #     weight_dict.update({k:state_dict[k]})
            #     if k not in state_dict.keys():
            #         print(k)
            # print(len(weight_dict))
            # print(len(state_dict))
            # print(len(self.model.state_dict()))
            # exit()
            self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.args.device)

    @torch.no_grad()
    def inference(self, fasta_lines, num_cycles):
        """
        Paramters
        ----------------------------------
        fasta_lines: []
            emulates a readlines(fasta_file), but is in this format to improve performance.
            list of strings where each even index elem corresponds to "> [SOME FASTA ID]"
                                  each odd index elem corresponds to "[AA String]"

        Return
        ----------------------------------
        node_repr: [num_mols, 16, num_residues, 384]
        edge_repr: [num_mols, num_residues, num_residues, 128]
        """
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        # logging.info(f"Reading {input_file}")

        node_results = []
        edge_results = []
        for i, input_data in enumerate(
                pipeline.fasta2inputs(
                    fasta_lines,
                    num_pseudo_msa=self.args.num_pseudo_msa,
                    device=self.args.device,
                    mask_rate=self.args.pseudo_msa_mask_rate,
                    num_cycle=num_cycles,
                )
        ):
            # logging.info(f"Predicting {i + 1}th chain in {input_file}")
            # logging.info(
            #     f"{len(input_data[0]['p_msa'][0])} residues in this chain."
            # )
            ts = time.time()
            # try:
            edge_repr, node_repr = self.model(
                    input_data,
                    predict_with_confidence=True,
                    fwd_cfg=self.forward_config
                )
            node_results.append(node_repr.cpu())
            edge_results.append(edge_repr.cpu())

            # save_file = "embeddings-" + save_path.split("/")[-1][:-4]
            # out_path = os.path.join(output_dir, save_file)

            # with open(out_path, "w+b") as f:
            #     cPickle.dump({"edge_repr": edge_repr, "node_repr": node_repr}, f)

            # except RuntimeError as e:
            #     logging.info(f"Failed to generate {save_path} due to {e}")
            #     logging.info(f"Skipping...")
            #     continue
            # logging.info(f"Finished prediction in {time.time() - ts:.2f} seconds.")
            # logging.info(f"Node and edge embeddings saved to {out_path}.")

            # logging.info(f"Saving prediction to {save_path}")
            # pipeline.save_pdb(
            #     pos14=output["final_atom_positions"],
            #     b_factors=output["confidence"] * 100,
            #     sequence=input_data[0]["p_msa"][0],
            #     mask=input_data[0]["p_msa_mask"][0],
            #     save_path=save_path,
            #     model=0
            # )
            # logging.info(f"Saved")
            # del output
            del edge_repr
            del node_repr
            gc.collect()
            torch.cuda.empty_cache()
            

        return edge_results, node_results

# =============================================================================
# Tests
# =============================================================================
if __name__ == '__main__':
    main()
