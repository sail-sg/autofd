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
from typing import Tuple
from functools import update_wrapper
from jax.tree_util import tree_flatten, tree_map, tree_unflatten
import inspect
import types
import numpy as np

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
    if callable(f):
      return tuple(
        cls.from_annotation(p.annotation)
        for p in inspect.signature(f).parameters.values()
      )
    elif hasattr(f, "shape"):
      shape = f.shape
    elif hasattr(f, "aval"):
      shape = f.aval.shape
    else:
      raise ValueError(f"Unknown f: {f} of type {type(f)}")
    return tuple(d.spec for d in shape[1:])

  @classmethod
  def from_ret(cls, f):
    if callable(f):
      return cls.from_annotation(inspect.signature(f).return_annotation)
    elif hasattr(f, "shape"):
      shape = f.shape
    elif hasattr(f, "aval"):
      shape = f.aval.shape
    return shape[0].spec

  @classmethod
  def from_value(cls, v):
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

  @classmethod
  def to_parameter(cls, spec_tree, name):
    return inspect.Parameter(
      name,
      kind=inspect.Parameter.POSITIONAL_ONLY,
      annotation=cls.to_annotation(spec_tree),
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
    args = self.shape[1:]
    return f"{','.join(map(str, args))}->{ret}".replace(" ", "")

  @property
  def num_args(self):
    return len(self.shape) - 1

  @property
  def arg_spec(self):
    return tuple(d.spec for d in self.shape[1:])

  @property
  def ret_spec(self):
    return self.shape[0].spec


jax.core.raise_to_shaped_mappings[GeneralArray
                                 ] = lambda aval, weak_type: GeneralArray(
                                   aval.shape,
                                   dtype=aval.dtype,
                                   weak_type=weak_type,
                                   named_shape=aval.named_shape,
                                 )


class function:
  """Wrapper for a function, to add some operator overloading.
  """

  def __init__(self, f, arg_spec=None, ret_spec=None):
    update_wrapper(self, f)
    self.f = f
    if arg_spec and not isinstance(arg_spec, (tuple, list)):
      raise ValueError(
        "arg_spec must be of Tuple[Pytree[Spec]], "
        f"got {arg_spec} of type {type(arg_spec)}"
      )
    if ret_spec and not isinstance(ret_spec, (tuple, list, Spec)):
      raise ValueError(
        "ret_spec must be a Union[Spec, Pytree[Spec]], "
        f"got {ret_spec} of type {type(ret_spec)}"
      )
    self.arg_spec = arg_spec or SpecTree.from_args(self.f)
    self.ret_spec = ret_spec or SpecTree.from_ret(self.f)

  @property
  def shape(self):
    return (
      Ret(self.ret_spec),
      *(Arg(s, f"arg{i}") for i, s in enumerate(self.arg_spec))
    )

  @property
  def num_args(self):
    return len(self.arg_spec)

  def __call__(self, *args):
    # TODO: can we use overload call to provide multiple functionality
    # e.g. f(*args) for evaluating the function
    # but when other functions are passed f(*gs), it triggers function
    # composition.
    return self.f(*args)


def with_spec(arg_spec=None, ret_spec=None):

  def decorator(f):
    return function(f, arg_spec=arg_spec, ret_spec=ret_spec)

  return decorator


# Extend the original dimension system with GeneralDim,
# to denote the inputs/output of functions.
class GeneralDim:
  """GeneralDim class, denotes either the input / output
  of a jaxtyping annotated python function.
  """

  def __init__(self, spec, name=None):
    self.spec = spec
    self.name = name

  def __eq__(self, other):
    # don't compare the name
    if isinstance(other, GeneralDim):
      return self.spec == other.spec
    else:
      return False

  def __gt__(self, other):
    return True

  def __ge__(self, other):
    return True

  def __repr__(self) -> str:
    return self.spec.__repr__()

  def __hash__(self) -> str:
    return hash(repr(self))

  def evaluate(self, env):
    return self

  def dimension_as_value(self) -> int:
    pass


def Arg(spec, name):
  return GeneralDim(spec, name=name)


def Ret(spec):
  return GeneralDim(spec)


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
  return dummy_array(f.arg_spec)


def random_input(key, f):
  return tree_map(
    lambda s: jax.random.normal(key, s.shape, dtype=s.dtype), f.arg_spec
  )


def dummy_array(spec_tree):
  return tree_map(
    lambda spec: jnp.zeros(spec.shape, dtype=spec.dtype),
    spec_tree,
  )


def jacobian_spec(f, argnums=0, has_aux=False):
  i_spec = f.arg_spec
  o_spec = f.ret_spec
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


# register with jax, so that when a python function is used for tracing,
# it is converted to an abstract ShapedArray type, that has infinite dimensions.
jax.core.pytype_aval_mappings[types.FunctionType] = function_to_aval
jax.interpreters.xla.pytype_aval_mappings[types.FunctionType] = function_to_aval
jax._src.api_util._shaped_abstractify_handlers[types.FunctionType
                                              ] = function_to_aval
jax._src.dtypes.python_scalar_dtypes[types.FunctionType] = np.dtype("float32")

# register function class
jax.core.pytype_aval_mappings[function] = function_to_aval
jax.interpreters.xla.pytype_aval_mappings[function] = function_to_aval
jax._src.api_util._shaped_abstractify_handlers[function] = function_to_aval
jax._src.dtypes.python_scalar_dtypes[function] = np.dtype("float32")
