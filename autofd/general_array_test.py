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

import jax
import jax.numpy as jnp
from jaxtyping import Float32, Array
from typing import Tuple
from absl.testing import absltest, parameterized
from autofd.general_array import (
  Arg,
  Ret,
  Spec,
  function,
  jacobian_spec,
  SpecTree,
  dummy_input,
)


@function
def function0(x: Float32[Array, "3 5"]) -> Float32[Array, ""]:
  return x.sum()


@function
def function1(x: Float32[Array, "3 5"]) -> Float32[Array, "3"]:
  return x.sum(1)


@function
def function2(
  x: Float32[Array, "3 5"]
) -> Tuple[Float32[Array, "3"], Float32[Array, "5"]]:
  return x.sum(1), x.sum(0)


@function
def function3(
  x: Float32[Array, "3 5"],
  y: Float32[Array, "3 5"],
) -> Tuple[Float32[Array, "3"], Float32[Array, "5"]]:
  return x.sum(1), y.sum(0)


@function
def function4(
  x: Float32[Array, "3 5"],
  y: Float32[Array, "3 5"],
) -> Tuple[Tuple[Float32[Array, "3"], Float32[Array, ""]], Float32[Array, "5"]]:
  return (x.sum(1), y.sum()), x.sum(0)


class _TestGeneralArray(parameterized.TestCase):

  def test_spec(self):
    self.assertEqual(
      function0.shape,
      (Ret(Spec((), jnp.float32)), Arg(Spec((3, 5), jnp.float32)))
    )
    self.assertEqual(
      function1.shape,
      (Ret(Spec((3,), jnp.float32)), Arg(Spec((3, 5), jnp.float32)))
    )
    self.assertEqual(
      function2.shape, (
        Ret((Spec((3,), jnp.float32), Spec((5,), jnp.float32))
           ), Arg(Spec((3, 5), jnp.float32))
      )
    )

  @parameterized.parameters(
    (function0, 0, False),
    (function1, 0, False),
    (function2, 0, True),
    (function2, 0, False),
    (function3, 0, True),
    (function3, 0, False),
    (function3, 1, True),
    (function3, 1, False),
    (function3, (0, 1), True),
    (function3, (0, 1), False),
    (function4, 0, True),
    (function4, 0, False),
    (function4, 1, True),
    (function4, 1, False),
    (function4, (0, 1), True),
    (function4, (0, 1), False),
  )
  def test_jacobian(self, f, argnums, has_aux):
    jac_spec1 = jacobian_spec(f, argnums=argnums, has_aux=has_aux)
    jac = jax.jacrev(f, argnums=argnums, has_aux=has_aux)(*dummy_input(f))
    jac_spec2 = SpecTree.from_value(jac)
    self.assertEqual(jac_spec1, jac_spec2)


if __name__ == "__main__":
  absltest.main()
