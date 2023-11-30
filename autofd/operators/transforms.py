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
import jax.linear_util as lu


@lu.transformation
def permute_args(perm, *args):
  inv_perm = tuple(np.argsort(perm))
  f_args = yield (args[i] for i in inv_perm), {}
  yield f_args


@lu.transformation
def return_at(i, *args):
  f_args = yield args, {}
  yield f_args[i]


@lu.transformation
def compose(gs, num_args, share_inputs, *args):
  i = 0
  if share_inputs:
    fargs = tuple(g(*args) for g in gs)
  else:
    fargs = []
    for g, na in zip(gs, num_args):
      fargs.append(g(*args[i:i + na]))
      i += na
    fargs = tuple(fargs)
  fgs_args = yield fargs, {}
  yield fgs_args


@lu.transformation
def broadcast_arg(shape, argnums, *args):
  args_shape = shape[1:]
  assert len(args_shape) == len(args)
  # assert tuple(Arg(Spec(a.shape, a.dtype)) for a in args) == args_shape
  f_args = yield (args[i] for i in argnums), {}
  yield f_args
