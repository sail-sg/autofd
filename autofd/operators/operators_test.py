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
from jax.tree_util import tree_map
from jaxtyping import Float32, Array
from typing import Tuple
import numpy as np
from functools import partial
from absl.testing import absltest, parameterized
from autofd.general_array import (
  SpecTree,
  function,
  dummy_input,
  random_input,
)
from autofd.operators.operators import (
  concat,
  split,
  compose,
  nabla,
  integrate,
  zip_functions,
  linear_transpose,
  pack_args,
  unpack_args,
  linearize,
  unary_funcs,
  _unary_compose,
  add,
  sub,
)


def _assert_tree_array_equal(a, b):
  tree_map(lambda x, y: np.testing.assert_array_equal(x, y), a, b)


def _assert_tree_array_almost_equal(a, b):
  tree_map(lambda x, y: np.testing.assert_array_almost_equal(x, y), a, b)


# Test the concat operator


@function
def concat_f(x: Float32[Array, "3 5"]) -> Float32[Array, "3"]:
  return jnp.sum(x, axis=-1)


@function
def concat_f1(x: Float32[Array, "3 5"]) -> Float32[Array, ""]:
  return jnp.max(x)


@function
def concat_f2(x: Float32[Array, "3 5"]) -> Float32[Array, "5"]:
  return x.min(axis=0)


@function
def concat_f3(
  x: Float32[Array, "3 5"],
) -> Tuple[Float32[Array, "3"], Float32[Array, ""], Float32[Array, "5"]]:
  return (x[:, 0], x[0, 0], x[0, :])


class _TestConcatSplit(absltest.TestCase):

  def test_concat_split(self):
    f, f1, f2 = (concat_f, concat_f1, concat_f2)
    f012 = concat(f, f1, f2)
    x = jax.random.uniform(jax.random.PRNGKey(0), (3, 5))
    out = f012(x)
    expected = (f(x), f1(x), f2(x))
    for o, e in zip(out, expected):
      np.testing.assert_array_equal(o, e)

    ff, ff1, ff2 = jax.linear_transpose(concat, f, f1, f2)(f012)
    out = (ff(x), ff1(x), ff2(x))
    for o, e in zip(out, expected):
      np.testing.assert_array_equal(o, e)

    f, f1, f2 = split(f012)
    ff012 = jax.linear_transpose(split, f012)([f, f1, f2],)[0]
    out1, out2 = f012(x), ff012(x)
    for o1, o2 in zip(out1, out2):
      np.testing.assert_array_equal(o1, o2)

  def test_concat_split_vjp(self):
    """This test depends on modification on jax.
    specifically, we need jax._src.dtypes.dtype to support function input.
    """
    f, f1, f2, f3 = (concat_f, concat_f1, concat_f2, concat_f3)
    f012, vjp = jax.vjp(concat, f, f1, f2)
    ff1, ff2, ff3 = vjp(f3)
    x = jax.random.uniform(jax.random.PRNGKey(0), (3, 5))
    np.testing.assert_array_equal(ff1(x), x[:, 0])
    np.testing.assert_array_equal(ff2(x), x[0, 0])
    np.testing.assert_array_equal(ff3(x), x[0, :])

    a, b = jax.vjp(split, f012)
    (ff, ff1, ff2), vjp = jax.vjp(split, f012)
    ff012 = vjp([ff, ff1, ff2])[0]

    for o1, o2 in zip(ff012(x), f012(x)):
      np.testing.assert_array_equal(o1, o2)


@function
def linear_transpose_f1(
  x: Float32[Array, "3 5"], /
) -> Tuple[Float32[Array, "3"], Float32[Array, ""], Float32[Array, "5"]]:
  return (x[:, 0], x[0, 0], x[0, :])


class _TestLinearTranspose(absltest.TestCase):

  def test_linear_transpose(self):
    t = linear_transpose(linear_transpose_f1)

    @function
    def dummy_t(
      x: Tuple[Float32[Array, "3"], Float32[Array, ""], Float32[Array, "5"]], /
    ) -> Tuple[Float32[Array, "3 5"]]:
      pass

    self.assertEqual(dummy_t.shape, t.shape)
    t(*dummy_input(t))
    # self.assertTrue(isinstance(o, Tuple[Float32[Array, "3 5"]]))

    tt = jax.linear_transpose(linear_transpose, linear_transpose_f1)(t)[0]
    self.assertEqual(linear_transpose_f1.shape, tt.shape)
    tt(*dummy_input(tt))

  def test_linear_transpose_fd(self):
    f1 = linear_transpose_f1
    tf1, vjp = jax.vjp(linear_transpose, f1)
    f1_prime = vjp(tf1)[0]
    self.assertEqual(f1_prime.shape, f1.shape)


@function
def zip_f1(x: Float32[Array, "3 5"]) -> Float32[Array, "3"]:
  return jnp.sum(x, axis=-1)


@function
def zip_f2(x: Float32[Array, "3 5"]) -> Float32[Array, "5"]:
  return jnp.sum(x, axis=0)


class _TestZipFunction(absltest.TestCase):

  def test_zip_functions(self):
    a = zip_functions(zip_f1)

    @function
    def dummy(x: Float32[Array, "3 5"], /) -> Tuple[Float32[Array, "3"]]:
      pass

    self.assertEqual(a.shape, dummy.shape)


# Test the compose operator


@function
def compose_g1(x: Float32[Array, "3 5"]) -> Float32[Array, "3"]:
  return jnp.sum(x, axis=-1)


@function
def compose_g2(x: Float32[Array, "3 5"]) -> Float32[Array, ""]:
  return jnp.max(x)


@function
def compose_f1(
  x: Float32[Array, "3"],
  y: Float32[Array, ""],
) -> Float32[Array, ""]:
  return jnp.min(x) + y


@function
def compose_g3(
  x: Float32[Array, "3 5"],
  y: Float32[Array, "3 5"],
) -> Float32[Array, "3"]:
  return jnp.sum(x, axis=-1) + jnp.max(y, axis=-1)


@function
def compose_g4(
  x: Float32[Array, "3 5"],
  y: Float32[Array, "3 5"],
) -> Tuple[Float32[Array, "3"], Float32[Array, ""]]:
  return (jnp.sum(x, axis=-1), jnp.max(y))


@function
def compose_f2(
  x: Tuple[Float32[Array, "3"], Float32[Array, ""]],
  y: Float32[Array, ""],
) -> Float32[Array, ""]:
  x, z = x
  return jnp.min(x) + y - z


class _TestCompose(absltest.TestCase):

  def setUp(self):
    super().setUp()

  def test_shape(self):
    pass

  def test_compose_f1_g1_g2(self):
    g1, g2, f1 = (compose_g1, compose_g2, compose_f1)
    x = jax.random.normal(jax.random.PRNGKey(12), (3, 5))
    f = compose(f1, g1, g2)
    y = f((x,), (x,))
    self.assertEqual(y, x.sum(-1).min() + x.max())

  def test_compose_f1_g3_g2(self):
    g2, g3, f1 = (compose_g2, compose_g3, compose_f1)
    x = jax.random.normal(jax.random.PRNGKey(12), (3, 5))
    y = jax.random.normal(jax.random.PRNGKey(42), (3, 5))
    f = compose(f1, g3, g2)
    z = f((x, y), (x,))
    self.assertEqual(z, (x.sum(-1) + y.max(-1)).min() + x.max())

  def test_compose_f2_g4_g2(self):
    g2, g4, f2 = (compose_g2, compose_g4, compose_f2)
    x = jax.random.normal(jax.random.PRNGKey(12), (3, 5))
    y = jax.random.normal(jax.random.PRNGKey(42), (3, 5))
    f = compose(f2, g4, g2)
    z = f((x, y), (x,))
    self.assertEqual(z, x.sum(-1).min() + x.max() - y.max())

  def test_compose_jvp_type(self):
    g2, g4, f2 = (compose_g2, compose_g4, compose_f2)
    primal_out, tangent_out = jax.jvp(compose, (f2, g4, g2), (f2, g4, g2))
    self.assertEqual(tangent_out.shape, primal_out.shape)

  def test_compose_vjp(self):

    @function
    def f(x: Float32[Array, ""]) -> Float32[Array, ""]:
      return x * 2

    @function
    def g(x: Float32[Array, ""]) -> Float32[Array, ""]:
      return x**2

    @function
    def o(x: Float32[Array, ""]) -> Float32[Array, ""]:
      return x**2 * 2

    def comp_f(g):
      return compose(f, g)

    def vjp_comp_f(g, o):
      out, vjp = jax.vjp(comp_f, g)
      return vjp(o)[0]

    out, vjp = jax.vjp(comp_f, g)
    gg = vjp(out)[0]
    gg(*dummy_input(gg))

  def test_euler_lagrange(self):

    @function
    def multiply_f_nablaf(f: Float32[Array, ""], nablaf: Float32[Array, ""],
                          /) -> Float32[Array, ""]:
      return jnp.multiply(f, nablaf)

    @function
    def sum_square_f_nablaf(
      f: Float32[Array, ""], nablaf: Float32[Array, ""], /
    ) -> Float32[Array, ""]:
      return jnp.add(f**2, nablaf**2)

    @function
    def square_f(f: Float32[Array, ""], nablaf: Float32[Array, ""],
                 /) -> Float32[Array, ""]:
      return nablaf**2

    integrand = sum_square_f_nablaf

    def I(f):
      return integrate(compose(integrand, f, nabla(f), share_inputs=True))

    # do this again with euler lagrange equation
    def DIDf(f):

      def func(x):
        # D I / D f = p I / p f - d/dx ( p I / p f')
        pIpf = jax.grad(integrand, argnums=0)(f(x), nabla(f)(x))
        ddx_pIpnablaf = jax.grad(
          lambda x: jax.grad(integrand, argnums=1)(f(x), nabla(f)(x))
        )(
          x
        )
        return pIpf - ddx_pIpnablaf

      return func

    # evaluate the functional on sin function
    @function
    def sin(x: Float32[Array, ""], /) -> Float32[Array, ""]:
      return jnp.sin(x)

    fd1 = jax.grad(I)(sin)
    fd2 = DIDf(sin)
    x = jax.random.normal(jax.random.PRNGKey(12), (10,))
    np.testing.assert_array_equal(jax.vmap(fd1)(x), jax.vmap(fd2)(x))


# Test add
@function
def add_f1(
  x: Float32[Array, "3 5"],
  y: Float32[Array, "3 5"],
) -> Tuple[Float32[Array, "3"], Float32[Array, ""]]:
  return jnp.min(x, axis=-1), jnp.min(y)


@function
def add_f2(
  x: Float32[Array, "3 5"],
  y: Float32[Array, "3 5"],
) -> Tuple[Float32[Array, "3"], Float32[Array, ""]]:
  return jnp.sum(x, axis=-1), jnp.max(y)


class _TestBinaryFunctions(absltest.TestCase):

  def test_add(self):
    f1, f2 = (add_f1, add_f2)
    f12 = add(f1, f2)
    x, y = (
      jax.random.normal(jax.random.PRNGKey(12), (3, 5)),
      jax.random.normal(jax.random.PRNGKey(42), (3, 5)),
    )
    tree_map(
      np.testing.assert_array_equal,
      f12(x, y),
      tree_map(jnp.add, f1(x, y), f2(x, y)),
    )

  def test_sub(self):
    f1, f2 = (add_f1, add_f2)
    f12 = sub(f1, f2)
    x, y = (
      jax.random.normal(jax.random.PRNGKey(12), (3, 5)),
      jax.random.normal(jax.random.PRNGKey(42), (3, 5)),
    )
    tree_map(
      np.testing.assert_array_equal,
      f12(x, y),
      tree_map(jnp.subtract, f1(x, y), f2(x, y)),
    )


# Test the nabla operator


@function
def nabla_f(x: Float32[Array, "3 5"]) -> Float32[Array, "3"]:
  return jnp.sum(jnp.sin(x * 2), axis=-1)


@function
def nabla_f1(x: Float32[Array, "3 5"]) -> Float32[Array, ""]:
  return jnp.sum(jnp.sin(x * 2))


@function
def nabla_f2(
  x: Float32[Array, "3 5"]
) -> Tuple[Float32[Array, ""], Float32[Array, "3"]]:
  return jnp.sum(jnp.sin(x * 2)), jnp.sin(x)[:, 0]


@function
def nabla_f3(x: Float32[Array, ""]) -> Float32[Array, ""]:
  return jnp.sin(x)


nabla_fs = (nabla_f, nabla_f1, nabla_f2, nabla_f3)


class _TestNabla(parameterized.TestCase):

  @parameterized.parameters(*((f, i) for f in nabla_fs for i in range(1, 2)))
  def test_nabla(self, f, recursion=1):
    jac_fn1, jac_fn2 = f, f
    for _ in range(recursion):
      jac_fn1 = nabla(jac_fn1)
      jac_fn2 = jax.jacrev(jac_fn2)

    args = random_input(jax.random.PRNGKey(42), f)
    jac1, jac2 = jac_fn1(*args), jac_fn2(*args)
    tree_map(np.testing.assert_array_equal, jac1, jac2)

  @parameterized.parameters(*((f,) for f in nabla_fs))
  def test_transpose(self, f):
    jacf = nabla(f)
    ff = jax.linear_transpose(nabla, f)(jacf)[0]
    self.assertEqual(ff.arg_spec, f.arg_spec)
    self.assertEqual(ff.ret_spec, f.ret_spec)
    args = random_input(jax.random.PRNGKey(42), ff)
    out = ff(*args)
    self.assertEqual(SpecTree.from_value(out), ff.ret_spec)

  @parameterized.parameters(*((f,) for f in nabla_fs))
  def test_functional_derivative(self, f):
    nabla_f = nabla(f)
    _, nabla_vjp = jax.vjp(nabla, f)
    g = nabla_vjp(nabla_f)[0]
    self.assertEqual(g.arg_spec, f.arg_spec)

  def test_known_functions(self):

    @function
    def sin(x: Float32[Array, ""]) -> Float32[Array, ""]:
      return jnp.sin(x)

    nabla_sin = nabla(sin)
    xs = jax.random.normal(jax.random.PRNGKey(0), (100,))
    for x in xs:
      np.testing.assert_allclose(nabla_sin(x), jnp.cos(x))

    nabla_vjp = jax.vjp(nabla, sin)[1]
    nabla_t_sin = nabla_vjp(sin)[0]
    for x in xs:
      np.testing.assert_allclose(nabla_t_sin(x), -jnp.cos(x))


class _TestPackUnpack(absltest.TestCase):

  def test_pack_unpack(self):

    @function
    def f(
      a: Float32[Array, "3 5"],
      b: Float32[Array, "3 5"],
    ) -> Float32[Array, "3 5"]:
      return jnp.add(a, b)

    packed_f1 = pack_args(f)
    x = jax.random.normal(jax.random.PRNGKey(0), (3, 5))
    y = jax.random.normal(jax.random.PRNGKey(1), (3, 5))
    unpacked_f1 = unpack_args(packed_f1)
    unpacked_f2 = jax.linear_transpose(pack_args, f)(packed_f1)[0]
    packed_f2 = jax.linear_transpose(unpack_args, packed_f1)(f)[0]
    np.testing.assert_array_equal(packed_f1((x, y)), f(x, y))
    np.testing.assert_array_equal(packed_f2((x, y)), f(x, y))
    np.testing.assert_array_equal(unpacked_f1(x, y), f(x, y))
    np.testing.assert_array_equal(unpacked_f2(x, y), f(x, y))


class _TestLinearize(absltest.TestCase):

  def test_linearize(self):

    @function
    def f(
      a: Float32[Array, "3 5"],
      b: Float32[Array, "3 5"],
    ) -> Float32[Array, "3 5"]:
      return a**2 + jnp.sin(b)

    x, y = (
      jax.random.normal(jax.random.PRNGKey(0), (3, 5)),
      jax.random.normal(jax.random.PRNGKey(1), (3, 5)),
    )
    x_dot, y_dot = (
      jax.random.normal(jax.random.PRNGKey(2), (3, 5)),
      jax.random.normal(jax.random.PRNGKey(3), (3, 5)),
    )
    out0 = linearize(f, argnums=(0,))((x, y), (x_dot,))
    expect0 = 2 * x * x_dot
    out1 = linearize(f, argnums=(1,))((x, y), (y_dot,))
    expect1 = jnp.cos(y) * y_dot
    out2 = linearize(f)((x, y), (x_dot, y_dot))
    expect2 = 2 * x * x_dot + jnp.cos(y) * y_dot
    np.testing.assert_array_equal(out0, expect0)
    np.testing.assert_array_equal(out1, expect1)
    np.testing.assert_array_equal(out2, expect2)


@function
def overload_f1(
  a: Float32[Array, "3 5"],
  b: Float32[Array, "3 5"],
) -> Float32[Array, "3 5"]:
  return a**2 + jnp.sin(b)


@function
def overload_f2(
  a: Float32[Array, "3 5"],
  b: Float32[Array, "3 5"],
) -> Tuple[Float32[Array, "3 5"], Float32[Array, "3 5"]]:
  return (a**2, jnp.sin(b))


@function
def overload_f3(
  a: Float32[Array, "3 5"],
  b: Float32[Array, "3 5"],
) -> Tuple[Float32[Array, "5"], Float32[Array, "5"]]:
  return ((a**2).sum(0), jnp.sin(b).sum(0))


class _TestOperatorOverload(parameterized.TestCase):
  # TODO: test whether the abstract methods works.
  # normally it should because ShapedArray has those overloaded.

  def test_broadcast(self):
    f = overload_f2
    g = function(f)
    args = random_input(jax.random.PRNGKey(0), f)
    # tree_map
    fp11 = g + jnp.ones([3, 5])
    fp12 = g + jnp.ones([5])
    _assert_tree_array_equal(fp11(*args), tree_map(lambda x: x + 1, f(*args)))
    _assert_tree_array_equal(fp12(*args), tree_map(lambda x: x + 1, f(*args)))

  @parameterized.parameters(overload_f1, overload_f2, overload_f3)
  def test_add(self, f):
    g = function(f)
    f2 = g + f
    args = random_input(jax.random.PRNGKey(0), f2)
    _assert_tree_array_equal(f2(*args), tree_map(lambda x: 2 * x, f(*args)))
    fp1 = g + 1
    _assert_tree_array_equal(fp1(*args), tree_map(lambda x: x + 1, f(*args)))

  @parameterized.parameters(overload_f1, overload_f2, overload_f3)
  def test_sub(self, f):
    g = function(f)
    f2 = g + g
    f2subf = f2 - f
    fsubf2 = f - f2
    args = random_input(jax.random.PRNGKey(0), f2subf)
    np.testing.assert_array_equal(f2subf(*args), f(*args))
    np.testing.assert_array_equal(
      fsubf2(*args), tree_map(lambda x: -x, f(*args))
    )

  @parameterized.parameters(overload_f1, overload_f2, overload_f3)
  def test_neg(self, f):
    g = function(f)
    args = random_input(jax.random.PRNGKey(0), g)
    np.testing.assert_array_equal((-g)(*args), tree_map(lambda x: -x, g(*args)))

  @parameterized.parameters(overload_f1, overload_f2, overload_f3)
  def test_pow(self, f):
    f = overload_f1
    g = function(f)
    args = random_input(jax.random.PRNGKey(0), g)
    _assert_tree_array_almost_equal(
      jax.jit(g**3)(*args), tree_map(lambda x: x**3, g(*args))
    )

  @parameterized.parameters(overload_f1, overload_f2, overload_f3)
  def test_mul(self, f):
    g = function(f)
    args = random_input(jax.random.PRNGKey(0), g)
    np.testing.assert_array_equal(
      (g * f)(*args), tree_map(lambda x: x**2, f(*args))
    )
    np.testing.assert_array_equal(
      (f * g)(*args), tree_map(lambda x: x**2, f(*args))
    )

  @parameterized.parameters(overload_f1, overload_f2, overload_f3)
  def test_div(self, f):
    g = function(f)
    args = random_input(jax.random.PRNGKey(0), g)
    _assert_tree_array_almost_equal(
      jax.jit(g / f)(*args), tree_map(jnp.ones_like, f(*args))
    )
    _assert_tree_array_almost_equal(
      (f / g)(*args), tree_map(jnp.ones_like, f(*args))
    )


class _TestUnary(parameterized.TestCase):

  @parameterized.parameters(
    (u, f) for _, u in unary_funcs.items() for f in (overload_f1, overload_f2)
  )
  def test_unary_compose(self, u, f):
    u_f = partial(_unary_compose, u)(f)
    out0 = u_f(*random_input(jax.random.PRNGKey(0), u_f))
    out1 = tree_map(u, f(*random_input(jax.random.PRNGKey(0), f)))
    _assert_tree_array_equal(out0, out1)


if __name__ == "__main__":
  absltest.main()
