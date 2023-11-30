from autofd.general_array import function, Grid
from autofd.utils import visualize_jaxpr  # noqa
import autofd.operators as o
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float32
import sys
import time

sys.setrecursionlimit(10000)


@function
def f(x: Float32[Array, ""]) -> Float32[Array, ""]:
  return jnp.sin(4 * x * jnp.pi)


@function
def b(x: Float32[Array, ""]) -> Float32[Array, ""]:
  return jnp.sin(x * jnp.pi)


@function
def y(x: Float32[Array, ""]) -> Float32[Array, ""]:
  return jnp.cos(x * jnp.pi)


@function
def k(y: Float32[Array, ""], x: Float32[Array, ""]) -> Float32[Array, ""]:
  return jnp.sin(y) + jnp.cos(x)


nodes = jnp.linspace(0, 1, 100)
weights = jnp.ones((100,)) / 100
f.grid = Grid((nodes,), (weights,))
k.grid = Grid((nodes, nodes), (weights, weights))


def layer(k, b, x, activate=True):
  fyx = o.broadcast_arg(x, k, argnums=1)
  g = o.integrate(fyx * k, argnums=1) + b
  if activate:
    a = o.numpy.tanh(g)
    return a
  else:
    return g


def loss(params, x, y):
  k1, b1, k2, b2 = params
  o1 = layer(k1, b1, x, True)
  o2 = layer(k2, b2, o1, False)
  return o.integrate((o2 - y)**2)


def out(params, x, y):
  k1, b1, k2, b2 = params
  o1 = layer(k1, b1, x, True)
  o2 = layer(k2, b2, o1, False)
  return o2


def main():
  param = (k, b, k, b)
  l = loss(param, f, y)
  ls = [l]
  outs = [out(param, f, y)]
  ks = [param[0]]
  print(f"Initial loss: {l}")
  for _ in range(4):
    t1 = time.time()
    grad = jax.grad(loss)(param, f, y)
    param = jax.tree_util.tree_map(lambda x, dx: x - 0.1 * dx, param, grad)
    t2 = time.time()
    l = loss(param, f, y)
    ls.append(l)
    ot = out(param, f, y)
    outs.append(ot)
    ks.append(param[0])
    # uncomment for visualizing jaxpr of function k1
    # visualize_jaxpr(param[0])(0., 0.)
    print(f"Loss at step {_} is {l}, spending {t2 - t1}s")
  return ls, outs, ks


main()
