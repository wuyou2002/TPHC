# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
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
    model_type = "tphc-llama"

    def __init__(
        self,
        num_streams: int = 4,
        num_groups: int = 4,
        num_updates: int = 4,
        **kwargs
    ):
        """
        Initializes the MyLlamaConfig.

        Args:
            num_streams (int, optional): Number of residual streams. Defaults to 4.
            num_groups (int, optional): Number of attn/ffn groups. Defaults to 4.
            num_updates (int, optional): Number of streams updated per layer by
                PostMapping/ResMapping. Defaults to 4. Must equal num_groups,
                except in the dense regime num_updates == num_streams > num_groups
                (every layer updates all streams).
                (per-update-stream causal kernels) at the MLP output. Defaults to False.
            **kwargs: Additional keyword arguments passed to the parent LlamaConfig.
        """
        super().__init__(**kwargs)
        self.num_streams = num_streams
        self.num_groups = num_groups
        self.num_updates = num_updates

        if num_groups != num_updates:
            # G != U is only supported in the "dense write-back" regime G < U = N:
            # every layer updates ALL N groups (update_groups = full range), and
            # Pre/PostMapping use the legacy dense init (sliding read window /
            # full-coverage write gates). PostMapping's mixing matrix is
            # (num_updates, num_groups); full per-layer coverage of the U
            # update groups requires num_streams % num_groups == 0.
            if num_updates != num_streams:
                raise ValueError(
                    f"`num_groups` ({num_groups}) != `num_updates` ({num_updates}) is only "
                    f"supported when num_updates == num_streams ({num_streams}): PostMapping's mixing "
                    f"matrix is (num_updates, num_groups) and the legacy dense init assumes "
                    f"every layer updates all groups."
                )
            if num_streams % num_groups != 0:
                raise ValueError(
                    f"`num_streams` ({num_streams}) must be divisible by `num_groups` "
                    f"({num_groups}): the legacy PostMapping init opens one gate per "
                    f"(i + j*G) % N, covering all N groups only when G divides N."
                )

        # validation
        if self.num_attention_heads % num_groups != 0:
            raise ValueError(
                f"`num_attention_heads` ({self.num_attention_heads}) must be divisible by "
                f"`num_groups` ({num_groups})."
            )
        if self.num_key_value_heads % num_groups != 0:
            raise ValueError(
                f"Either `num_key_value_heads` ({self.num_key_value_heads}) must be divisible by "
                f"`num_groups` ({num_groups})."
            )
        if self.intermediate_size % num_groups != 0:
            raise ValueError(
                f"`intermediate_size` ({self.intermediate_size}) must be divisible by "
                f"`num_groups` ({num_groups})."
            )
