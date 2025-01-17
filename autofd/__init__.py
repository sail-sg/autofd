#!/usr/bin/env python3
# Copyright 2023 Garena Online Private Limited
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

try:
  import jax
  import optax

  def scale_by_learning_rate(
    learning_rate,
    *,
    flip_sign: bool = True,
  ):
    m = -1 if flip_sign else 1
    # avoid calling tracer
    if callable(learning_rate
               ) and not isinstance(learning_rate, jax.core.Tracer):
      return optax._src.transform.scale_by_schedule(
        lambda count: m * learning_rate(count)
      )
    return optax._src.transform.scale(m * learning_rate)

  optax._src.alias._scale_by_learning_rate = scale_by_learning_rate
except ImportError:
  print("optax not install, skip patching")

from . import operators  # noqa
from .general_array import SpecTree  # noqa
from .general_array import (
  Arg, Grid, Ret, Spec, dummy_array, dummy_input, dummy_output, function,
  is_function, num_args, random_input, with_spec, zeros_like
)

__all__ = [
  "Spec",
  "SpecTree",
  "Arg",
  "Ret",
  "Grid",
  "function",
  "with_spec",
  "num_args",
  "zeros_like",
  "is_function",
  "dummy_input",
  "dummy_output",
  "random_input",
  "dummy_array",
  "operators",
]

__version__ = "0.0.8"  # noqa
