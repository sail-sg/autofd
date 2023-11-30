import jax
import jax.numpy as jnp
from jaxtyping import Array, Float32
from autofd import function
import time


@function
def h(x: Float32[Array, ""]) -> Float32[Array, ""]:
  return jnp.sin(x)


@function
def f(x: Float32[Array, ""]) -> Float32[Array, ""]:
  return jnp.exp(x)


@function
def g(x: Float32[Array, ""]) -> Float32[Array, ""]:
  return jnp.tanh(x)


if __name__ == "__main__":

  def time_execution(depth):

    # functional to test
    def F(h):
      for _ in range(depth):
        h = f(h) + g(h)
      return h

    # Fh is h = f(h) + g(h) nested to depth times
    Fh = F(h)
    # time the execution
    t1 = time.time()
    jax.jit(Fh)(0.)
    t2 = time.time()
    return t2 - t1

  # execute once to warm up
  time_execution(1)

  # measure the execution time for a series of nest depths
  for i in range(30):
    print(f"depth: {i} took {time_execution(i)}")
