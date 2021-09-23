# -*- coding: utf-8 -*-
#
# Copyright (c) 2015 Cisco Systems, Inc. and others.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

try:
    import torch
except ImportError:
    pass

logger = logging.getLogger(__name__)


def get_disk_space_of_model(pytorch_module):
    """
    Returns the disk space of a pytorch module

    Args:
        pytorch_module: a pytorch neural network module derived from torch.nn.Module

    Returns:
        size (float): The size of model when dumped
    """
    filename = "temp.bin"
    torch.save(pytorch_module.state_dict(), filename)
    size = os.path.getsize(filename) / 1e6
    os.remove(filename)
    msg = f"Pytorch module will be dumped temporarily at {filename} in order to " \
          f"calculate its disk size."
    logger.debug(msg)
    return size


def get_num_weights_of_model(pytorch_module):
    """
    Returns the number of trainable and the total parameters in a pytorch module. Returning both
    helps to do a sanity check if any layers which are meant to be frozen are being trained or not.

    Args:
        pytorch_module: a pytorch neural network module derived from torch.nn.Module

    Returns:
        number_of_params (tuple): A tuple of number of params that are trainable and total number
            of params of the pytorch module
    """
    n_total = 0
    n_requires_grad = 0
    for param in list(pytorch_module.parameters()):
        t = param.numel()
        n_total += t
        if param.requires_grad:
            n_requires_grad += t
    return n_requires_grad, n_total
