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
"""Core functions."""
import jax
import jax.numpy as jnp
import jaxtyping
from jaxtyping import Array
from jax.tree_util import (
  tree_flatten,
  tree_map,
  tree_unflatten,
  tree_structure,
)
from typing import Tuple
import jax.extend.linear_util as lu
import inspect
import types
import numpy as np
from dataclasses import dataclass
from .utils import weak_lru

# this is to trick jax to return dtype=float32 for functions
# TODO: generalize dtype for general arrays (aka functions).
MetaAbstractArray = jaxtyping._array_types._MetaAbstractArray  # noqa

dtype_to_jaxtyping = {
  "uint8": jaxtyping.UInt8,
  "uint16": jaxtyping.UInt16,
  "uint32": jaxtyping.UInt32,
  "uint64": jaxtyping.UInt64,
  "int8": jaxtyping.Int8,
  "int16": jaxtyping.Int16,
  "int32": jaxtyping.Int32,
  "int64": jaxtyping.Int64,
  "bfloat16": jaxtyping.BFloat16,
  "float16": jaxtyping.Float16,
  "float32": jaxtyping.Float32,
  "float64": jaxtyping.Float64,
  "complex64": jaxtyping.Complex64,
  "complex128": jaxtyping.Complex128,
  "bool": jaxtyping.Bool,
  "bool_": jaxtyping.Bool,
}


class Spec:

  def __init__(self, shape, dtype):
    self.shape = shape
    self.dtype = np.dtype(dtype)

  def __repr__(self) -> str:
    return f"{self.dtype.char}{list(self.shape)}"

  def __eq__(self, other):
    if isinstance(other, Spec):
      return self.shape == other.shape and self.dtype == other.dtype
    else:
      return False

  @staticmethod
  def from_annotation(ann):
    if isinstance(ann, MetaAbstractArray):
      if len(ann.dtypes) > 1:
        raise ValueError(
          f"Currently only support single dtype, got {ann}, "
          f"which has multiple dtypes of {ann.dtypes}."
        )
      return Spec(tuple(_normalize_dim(dim) for dim in ann.dims), ann.dtypes[0])
    else:
      raise ValueError(f"Unknown annotation: {ann} of type {type(ann)}")


class SpecTree:
  """Utility functions for converting between Pytree[Spec] and Annotation.
  """

  @staticmethod
  def from_annotation(ann):
    if hasattr(ann, "_name") and ann._name == "Tuple":
      return tuple(SpecTree.from_annotation(a) for a in ann.__args__)
    else:
      return Spec.from_annotation(ann)

  @classmethod
  def from_args(cls, f):
    if hasattr(f, "arg_spec"):
      return f.arg_spec
    elif callable(f):
      return tuple(
        cls.from_annotation(p.annotation)
        for p in inspect.signature(f).parameters.values()
      )
    elif hasattr(f, "aval") and isinstance(f.aval, GeneralArray):
      return f.aval.arg_spec
    elif hasattr(f, "shape") and isinstance(f.shape[0], GeneralDim):
      return tuple(d.spec for d in f.shape[1:])
    else:
      raise ValueError(f"Invalid function: {f} of type {type(f)}")

  @classmethod
  def from_ret(cls, f):
    if hasattr(f, "ret_spec"):
      return f.ret_spec
    elif callable(f):
      return cls.from_annotation(inspect.signature(f).return_annotation)
    elif hasattr(f, "aval") and isinstance(f.aval, GeneralArray):
      return f.aval.ret_spec
    elif hasattr(f, "shape") and isinstance(f.shape[0], GeneralDim):
      return f.shape[0].spec
    else:
      raise ValueError(f"Invalid function: {f} of type {type(f)}")

  @classmethod
  def from_value(cls, v):
    if isinstance(v, float):
      return Spec((), jnp.float32)
    return tree_map(lambda x: Spec(x.shape, x.dtype), v)

  @classmethod
  def to_annotation(cls, spec_tree):
    if isinstance(spec_tree, tuple):
      return Tuple[*(cls.to_annotation(s) for s in spec_tree)]
    elif isinstance(spec_tree, Spec):
      shape_str = " ".join(map(str, spec_tree.shape))
      key = spec_tree.dtype.name
      return dtype_to_jaxtyping[key][Array, shape_str]
    else:
      raise ValueError(
        f"Unknown spec_tree: {spec_tree} of type {type(spec_tree)}"
      )


class GeneralArray(jax.core.ShapedArray):

  def __init__(
    self, shape, dtype=jnp.float32, weak_type=True, named_shape=None
  ):
    # TODO: currently setting dtype to types.FunctionType will cause
    # weird error.
    super().__init__(
      shape, dtype=dtype, weak_type=weak_type, named_shape=named_shape
    )

  def update(self, shape=None, dtype=None, weak_type=None, named_shape=None):
    sa = super().update(shape, dtype, weak_type, named_shape)
    return GeneralArray(sa.shape, sa.dtype, sa.weak_type, sa.named_shape)

  def at_least_vspace(self):
    sa = super().at_least_vspace()
    return GeneralArray(sa.shape, sa.dtype, sa.weak_type, sa.named_shape)

  def str_short(self, short_dtypes=True):
    ret = self.shape[0]
    arg_str = ",".join(
      map(lambda x: "{...}" if isinstance(x, dict) else str(x), self.arg_spec)
    )
    return f"{arg_str}->{ret}".replace(" ", "")

  @property
  def num_args(self):
    return len(self.shape) - 1

  @property
  def arg_spec(self):
    return tuple(d.spec for d in self.arg)

  @property
  def ret_spec(self):
    return self.ret.spec

  @property
  def arg(self):
    return self.shape[1:]

  @property
  def ret(self):
    return self.shape[0]


jax.core.raise_to_shaped_mappings[GeneralArray
                                 ] = lambda aval, weak_type: GeneralArray(
                                   aval.shape,
                                   dtype=aval.dtype,
                                   weak_type=weak_type,
                                   named_shape=aval.named_shape,
                                 )


class operator:
  """Decorating a WrappedFun with this class makes it cached.
  After this decoration, no more wrapper can be appended to the transforms
  stack of the original WrappedFun f.

  When applying lu.transformations on this class, a new WrappedFun is created
  with this class as the core function.
  """

  def __init__(self, f):
    assert isinstance(f, lu.WrappedFun)
    self.f = f

  @weak_lru(maxsize=128, typed=False)
  def call_flatten_args_cached(self, tree_def, *args):
    args, kwargs = tree_unflatten(tree_def, args)
    return self.f.call_wrapped(*args, **kwargs)

  def __call__(self, *args, **kwargs):
    flat_args, tree_def = tree_flatten((args, kwargs))
    if all(map(lambda a: isinstance(a, jax.core.Trace), flat_args)):
      return self.call_flatten_args_cached(tree_def, *flat_args)
    else:
      return self.f.call_wrapped(*args, **kwargs)

  def wrap(self, gen, gen_static_args, out_store):
    return lu.WrappedFun(
      self, ((gen, gen_static_args),), (out_store,), (), None, None
    )

  def __repr__(self):
    return f"{self.f}"

  def __hash__(self):
    return hash(self.f)


class function:
  """Wrapper for a function, to add some operator overloading.
  """

  def __init__(self, f):
    if not isinstance(f, lu.WrappedFun):
      self.shape = (Ret(SpecTree.from_ret(f)), *map(Arg, SpecTree.from_args(f)))
      f = lu.wrap_init(f)
    self.f = f
    self._f_hash = hash(self.f)
    self._jaxpr = None
    self._arrays = {}

  @property
  def grid(self):
    if "grid" not in self._arrays:
      self._arrays["grid"] = Grid.none(len(self.arg))
    return self._arrays["grid"]

  @grid.setter
  def grid(self, g):
    self._arrays["grid"] = g

  def wrap(self, gen, gen_static_args, out_store):
    return lu.WrappedFun(
      self, ((gen, gen_static_args),), (out_store,), (), None, None
    )

  def __repr__(self) -> str:
    arg_str = ",".join(
      map(lambda x: "{...}" if isinstance(x, dict) else str(x), self.arg_spec)
    )
    return f"{arg_str} -> {self.ret_spec}"
    # return str(self.f)
    # return (f"{self.f} with signature "
    #         f"{arg_str} -> {self.ret_spec}")

  def __hash__(self):
    # as long as function call cache is concerned,
    # the hash is independent on self._arrays.
    return self._f_hash

  @weak_lru(maxsize=128, typed=False)
  def call_flatten_cached(self, tree_def, *args):
    return tree_flatten(self.f.call_wrapped(*tree_unflatten(tree_def, args)))[0]

  def _call(self, *args):
    out_def = tree_structure(self.ret_spec)
    flat_args, tree_def = tree_flatten(args)
    flat_args = tuple(flat_args)
    if all(map(lambda a: isinstance(a, jax.core.Tracer), flat_args)):
      return tree_unflatten(
        out_def, self.call_flatten_cached(tree_def, *flat_args)
      )
    else:
      # if evaluated with ArrayImpl
      if self._jaxpr is None:
        self._jaxpr = jax.make_jaxpr(
          lambda *args: self.call_flatten_cached(tree_def, *args)
        )(*flat_args)
        self._jaxpr_fun = jax.core.jaxpr_as_fun(self._jaxpr)
      return tree_unflatten(out_def, self._jaxpr_fun(*flat_args))

  @property
  def num_args(self):
    return len(self.shape) - 1

  @property
  def arg_spec(self):
    return tuple(a.spec for a in self.shape[1:])

  @property
  def ret_spec(self):
    return self.ret.spec

  @property
  def arg(self):
    return self.shape[1:]

  @property
  def ret(self):
    return self.shape[0]


def with_spec(arg_spec=None, ret_spec=None):

  def decorator(f):
    ff = function(lu.wrap_init(f))
    ff.shape = (Ret(ret_spec), *map(Arg, arg_spec))
    return ff

  return decorator


# Extend the original dimension system with GeneralDim,
# to denote the inputs/output of functions.
class GeneralDim:
  """GeneralDim class, denotes either the input / output
  of a jaxtyping annotated python function.
  """

  def __init__(self, spec):
    self.spec = spec

  def __eq__(self, other):
    if isinstance(other, GeneralDim):
      return self.spec == other.spec
    else:
      return False

  def __gt__(self, other):
    return True

  def __ge__(self, other):
    return True

  def __repr__(self) -> str:
    return repr(self.spec)

  def __hash__(self) -> str:
    return hash(repr(self.spec))

  def evaluate(self, env):
    return self

  def dimension_as_value(self) -> int:
    pass

  def to_shaped_array(self):
    return tree_map(
      lambda s: jax.core.ShapedArray(s.shape, dtype=s.dtype), self.spec
    )


def Arg(spec):
  return GeneralDim(spec)


def Ret(spec):
  return GeneralDim(spec)


def num_args(f):
  return getattr(f, "num_args", len(SpecTree.from_args(f)))


def ret(f):
  return Ret(SpecTree.from_ret(f))


def ret_like(v):
  return Ret(SpecTree.from_value(v))


# Utility functions for interoperation between
# jaxtyping, shape in pytree, and GeneralDim.
def _normalize_dim(dim):
  if isinstance(dim, (GeneralDim, int)):
    return dim
  elif isinstance(dim, jaxtyping._array_types._SymbolicDim):  # noqa
    return _normalize_dim(eval(dim))
  elif isinstance(dim, jaxtyping._array_types._FixedDim):  # noqa
    return dim.size
  else:
    raise ValueError(f"Unknown dim type: {dim} of type {type(dim)}")


# Convert a function to its abstract value.
def function_to_aval(f):
  """Convert a function to its abstract value."""
  if isinstance(f, types.FunctionType):
    raise ValueError(
      f"Please decorate the function {f} with @function or @with_spec."
    )
  elif isinstance(f, function):
    return GeneralArray(f.shape)


def is_function(f):
  # identify whether it is tracer first
  if hasattr(f, "aval"):
    return isinstance(f.aval, GeneralArray)
  elif callable(f):
    return True
  elif isinstance(f, GeneralArray):
    return True


def zeros_like(f):

  @with_spec(arg_spec=SpecTree.from_args(f), ret_spec=SpecTree.from_ret(f))
  def _zero(*args):
    return tree_map(jnp.zeros, SpecTree.from_ret(f))

  return _zero


def parameters(f):
  """Generate a tuple of `inspect.Parameter` from a function or
  the aval of a function.

  Args:
    f: A function or the aval of a function
  Returns:
    A tuple of `inspect.Parameter`.
  """
  spec_tree = SpecTree.from_args(f)
  return tuple(
    SpecTree.to_parameter(spec, f"arg{i}") for i, spec in enumerate(spec_tree)
  )


def return_annotation(f):
  """Get the return annotation from a function or the aval of a function.

  Args:
    f: A function or the aval of a function
  Returns:
    The return annotation.
  """
  return SpecTree.to_annotation(SpecTree.from_ret(f))


def signature(f):
  """Extract the signature from a function or the aval of a function.

  Args:
    f: A function or the aval of a function
  Returns:
    A `inspect.Signature` object.
  """
  return inspect.Signature(
    parameters(f), return_annotation=return_annotation(f)
  )


def dummy_input(f):
  return dummy_array(SpecTree.from_args(f))


def dummy_output(f):
  return dummy_array(SpecTree.from_ret(f))


def random_input(key, f):
  return tree_map(
    lambda s: jax.random.normal(key, s.shape, dtype=s.dtype),
    SpecTree.from_args(f)
  )


def dummy_array(spec_tree):
  return tree_map(
    lambda spec: jnp.zeros(spec.shape, dtype=spec.dtype),
    spec_tree,
  )


def jacobian_spec(f, argnums=0, has_aux=False):
  i_spec = SpecTree.from_args(f)
  o_spec = SpecTree.from_ret(f)
  if has_aux:
    if not isinstance(o_spec, tuple) or len(o_spec) != 2:
      raise ValueError("has_aux=True requires f to have 2 outputs.")
    o_spec, aux_spec = o_spec
  else:
    aux_spec = None
  o_flat, o_def = tree_flatten(o_spec)
  if isinstance(argnums, int):
    i_spec = i_spec[argnums]
  elif isinstance(argnums, (tuple, list)):
    i_spec = tuple(i_spec[i] for i in argnums)
  i_flat, i_def = tree_flatten(i_spec)
  jac_spec = tree_unflatten(
    o_def,
    list(
      tree_unflatten(
        i_def, list(Spec(o.shape + i.shape, i.dtype) for i in i_flat)
      ) for o in o_flat
    )
  )
  if has_aux:
    return jac_spec, aux_spec
  else:
    return jac_spec


@dataclass
class Grid:
  nodes: Tuple[jnp.ndarray]
  weights: Tuple[jnp.ndarray]

  @staticmethod
  def none(num):
    return Grid((None,) * num, (None,) * num)

  def take(self, indice):
    return Grid(
      tuple(map(self.nodes.__getitem__, indice)),
      tuple(map(self.weights.__getitem__, indice)),
    )

  def __add__(self, other):
    return Grid(self.nodes + other.nodes, self.weights + other.weights)

  def __radd__(self, other):
    if other == 0 or other is None:
      return self
    else:
      return other + self


# register with jax, so that when a python function is used for tracing,
# it is converted to an abstract ShapedArray type, that has infinite dimensions.
jax.core.pytype_aval_mappings[types.FunctionType] = function_to_aval
jax.interpreters.xla.pytype_aval_mappings[types.FunctionType] = function_to_aval
jax.interpreters.xla.canonicalize_dtype_handlers[types.FunctionType
                                                ] = lambda x: x
jax._src.api_util._shaped_abstractify_handlers[types.FunctionType
                                              ] = function_to_aval
jax._src.dtypes.python_scalar_dtypes[types.FunctionType] = np.dtype("float32")

# register function class
jax.core.pytype_aval_mappings[function] = function_to_aval
jax.interpreters.xla.pytype_aval_mappings[function] = function_to_aval
jax.interpreters.xla.canonicalize_dtype_handlers[function] = lambda x: x
jax._src.api_util._shaped_abstractify_handlers[function] = function_to_aval
jax._src.dtypes.python_scalar_dtypes[function] = np.dtype("float32")
