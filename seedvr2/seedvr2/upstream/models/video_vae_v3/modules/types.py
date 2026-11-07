# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from enum import Enum
from typing import Dict, Literal, NamedTuple, Optional
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

_receptive_field_t = Literal["half", "full"]
_inflation_mode_t = Literal["none", "tail", "replicate"]
_memory_device_t = Optional[Literal["cpu", "same"]]
_gradient_checkpointing_t = Optional[Literal["half", "full"]]
_selective_checkpointing_t = Optional[Literal["coarse", "fine"]]

class MemoryState(Enum):
    """
    State[Disabled]:        No memory bank will be enabled.
    State[Initializing]:    The model is handling the first clip, need to reset the memory bank.
    State[Active]:          There has been some data in the memory bank.
    State[Unset]:           Error state, indicating users didn't pass correct memory state in.
    """

    DISABLED = 0
    INITIALIZING = 1
    ACTIVE = 2
    UNSET = 3


class QuantizerOutput(NamedTuple):
    latent: torch.Tensor
    extra_loss: torch.Tensor
    statistics: Dict[str, torch.Tensor]


class CausalEncoderOutput(NamedTuple):
    latent: torch.Tensor
    posterior: Optional[DiagonalGaussianDistribution]


class CausalDecoderOutput(NamedTuple):
    sample: torch.Tensor
