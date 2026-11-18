# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library.  It has been modified from
# its original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" LLaMA model configuration"""
from typing import List, Optional

from transformers.models.llama.configuration_llama import LlamaConfig

class MyLlamaConfig(LlamaConfig):
    model_type = "mhc-llama"

    def __init__(
        self,
        num_streams: int = 4,
        temporal_aug: bool = False,
        temporal_kernel_sizes: Optional[List[int]] = None,
        gram_schmidt: bool = False,
        **kwargs
    ):
        """
        Initializes the MyLlamaConfig.

        Args:
            num_streams (int, optional): Number of residual streams (N). Defaults to 4.
            temporal_aug (bool, optional): Whether to apply temporal feature augmentation
                to the MLP output. Defaults to True.
            temporal_kernel_sizes (list, optional): Causal depthwise conv kernel sizes for
                temporal augmentation. Defaults to [4, 8, 12].
            gram_schmidt (bool, optional): Whether to Gram-Schmidt orthogonalize the
                temporal components. Defaults to True.
            **kwargs: Additional keyword arguments passed to the parent LlamaConfig.
        """
        super().__init__(**kwargs)
        self.num_streams = num_streams
        self.temporal_aug = temporal_aug
        self.temporal_kernel_sizes = list(
            [4, 8, 12] if temporal_kernel_sizes is None else temporal_kernel_sizes
        )
        self.gram_schmidt = gram_schmidt
