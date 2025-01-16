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
"""lu transformations"""

import numpy as np
import jax.extend.linear_util as lu


@lu.transformation2
def permute_args(f, perm, *args):
  inv_perm = tuple(np.argsort(perm))
  return f(*(args[i] for i in inv_perm))


@lu.transformation2
def return_at(f, i, *args):
  return f(*args)[i]


@lu.transformation2
def compose(f, gs, num_args, share_inputs, *args):
  i = 0
  if share_inputs:
    fargs = tuple(g(*args) for g in gs)
  else:
    fargs = []
    for g, na in zip(gs, num_args):
      fargs.append(g(*args[i:i + na]))
      i += na
    fargs = tuple(fargs)
  return f(*fargs)


@lu.transformation2
def broadcast_arg(f, shape, argnums, *args):
  args_shape = shape[1:]
  assert len(args_shape) == len(args)
  # assert tuple(Arg(Spec(a.shape, a.dtype)) for a in args) == args_shape
  return f(*(args[i] for i in argnums))
