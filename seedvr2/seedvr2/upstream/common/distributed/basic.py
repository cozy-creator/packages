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

"""
Distributed basic functions.
"""

import torch
import torch.distributed as dist


def get_global_rank() -> int:
    """
    Get the global rank, the global index of the GPU.
    """
    return 0


def get_local_rank() -> int:
    """
    Get the local rank, the local index of the GPU.
    """
    return 0


def get_world_size() -> int:
    """
    Get the world size, the total amount of GPUs.
    """
    return 1


def barrier_if_distributed(*args, **kwargs):
    """
    Synchronizes all processes if under distributed context.
    """
    if dist.is_initialized():
        return dist.barrier(*args, **kwargs)
