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

"""
# =============================================================================
# Imports
# =============================================================================
from typing import Dict, Union

import torch

from omegafold.utils.protein_utils import residue_constants
from omegafold.utils.protein_utils.aaframe import AAFrame
from omegafold.utils.protein_utils.functions import (
    bit_wise_not,
    create_pseudo_beta,
    get_norm,
    robust_normalize,
)
from omegafold.utils.torch_utils import (
    mask2bias,
    masked_mean,
    normalize,
    recursive_to,
)

# =============================================================================
# Constants
# =============================================================================
DATA = Dict[str, Union[str, bool, torch.Tensor, AAFrame]]
# =============================================================================
# Functions
# =============================================================================
# =============================================================================
# Classes
# =============================================================================
# =============================================================================
# Tests
# =============================================================================
if __name__ == "__main__":
    pass
