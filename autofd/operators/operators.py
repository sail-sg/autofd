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
"""A collection of operators."""

import types
from typing import Tuple
from functools import partial

import jax
import jax.numpy as jnp
from jax.tree_util import (
  tree_map,
  tree_flatten,
  tree_unflatten,
  tree_structure,
)
from jax._src.tree_util import broadcast_prefix
from jax.interpreters import ad
from jax._src import ad_util
from jax import core

from autofd.general_array import (
  Ret,
  Arg,
  Spec,
  SpecTree,
  GeneralArray,
  function,
  with_spec,
  jacobian_spec,
  dummy_array,
  dummy_output,
)


def _assert_same_input_signature(*fs):
  arg_specs = tuple(f.arg_spec for f in fs)
  for d in arg_specs[1:]:
    if d != arg_specs[0]:
      raise ValueError("All functions must have the same input signature.")


def _assert_same_output_signature(*fs):
  arg_specs = tuple(f.ret_spec for f in fs)
  for d in arg_specs[1:]:
    if d != arg_specs[0]:
      raise ValueError("All functions must have the same output signature.")


def _assert_composable(f, *gs):
  f_i_spec = f.arg_spec
  gs_o_spec = tuple(g.ret_spec for g in gs)
  if f_i_spec != gs_o_spec:
    raise ValueError(
      f"Cannot compose functions with incompatible shapes: "
      f"f requires: {f.arg_spec}, "
      f"while gs provides {tuple(g.ret_spec for g in gs)}"
    )


def _normalize_argnums(f, argnums):
  if argnums is None:
    argnums = tuple(range(f.num_args))
  elif isinstance(argnums, int):
    argnums = (argnums,)
  if isinstance(argnums, tuple):
    if len(set(argnums)) < len(argnums):
      raise ValueError("argnums must be unique.")
    elif any(map(lambda x: x >= f.num_args, argnums)):
      raise ValueError("argnums must be less than the number of arguments.")
  return argnums


def concat(*fs):
  """Concatenate functions.
  This requires all `*fs` to have the same input signature.
  Returns a function that returns a tuple of the outputs of the functions.

  Equivalent to:

  .. code-block:: python

    def concat(*fs):
      def concatenated_fs(*args):
        return tuple(f(*args) for f in fs)
      return concatenated_fs

  Args:
    *fs: functions to be concatenated.
  Returns:
    A function that returns a tuple of the outputs of the functions.
  """
  fs_o_spec = tuple(f.ret_spec for f in fs)

  @with_spec(arg_spec=fs_o_spec, ret_spec=fs_o_spec)
  def _concat_args(*args):
    return args

  return compose(
    _concat_args,
    *fs,
    share_inputs=True,
    f_type="linear",
  )


def split(f):
  return split_p.bind(f)


def _split_impl(f):
  if not isinstance(f.ret_spec, tuple):
    raise ValueError("Function must return a tuple.")

  def _split_i(i, *args, **kwargs):
    return f(*args, **kwargs)[i]

  fns = []
  for i, spec in enumerate(f.ret_spec):
    fns.append(function(partial(_split_i, i), f.arg_spec, spec))
  return tuple(fns)


split_p = core.Primitive("split")
split_p.def_impl(_split_impl)
split_p.multiple_results = True


@split_p.def_abstract_eval
def split_p_abstract_eval(f):
  ret_spec = f.ret_spec
  if not isinstance(ret_spec, tuple):
    raise ValueError("function must return a tuple.")
  return tuple(GeneralArray((Ret(s), *f.shape[1:])) for s in ret_spec)


def _split_transpose_rule(t, f):
  return (concat(*t),)


jax.interpreters.ad.deflinear2(split_p, _split_transpose_rule)
jax.interpreters.ad.primitive_transposes[split_p] = _split_transpose_rule


def zip_functions(*fs, share_inputs=False):
  arg_spec = tuple(f.ret_spec for f in fs)

  @with_spec(arg_spec, arg_spec)
  def _zipped(*args, **kwargs):
    return args

  return compose(
    _zipped,
    *fs,
    share_inputs=share_inputs,
    f_type="linear",
  )


def linear_transpose(f, *, argnums=None):
  """Returns the transpose of a linear function."""
  if argnums is not None and not isinstance(argnums, int):
    raise NotImplementedError(
      "Partial linear transposition only support a single integer for argnums."
    )
  return linear_transpose_p.bind(f, argnums=argnums)


def _linear_transpose_impl(f, *, argnums):
  i_spec = f.arg_spec
  o_spec = f.ret_spec

  if argnums is None:
    i_spec_t = (o_spec,)
    o_spec_t = i_spec
    param_names = ["cotangent"]
  else:
    i_spec_t = list(i_spec)
    i_spec_t[argnums] = o_spec
    o_spec_t = i_spec[argnums]
    param_names = list(f"arg{i}" for i in range(len(i_spec_t)))
    param_names[argnums] = "cotangent"

  @with_spec(arg_spec=tuple(i_spec_t), ret_spec=o_spec_t)
  def _linear_transposed(*args):
    if argnums is None:
      return jax.linear_transpose(f, *dummy_array(i_spec))(*args)
    else:
      cotangent = args[argnums]
      args = list(args)

      def partial_f(arg):
        args[argnums] = arg
        return f(*args)

      return jax.linear_transpose(partial_f,
                                  dummy_array(i_spec[argnums]))(cotangent)[0]

  return _linear_transposed


linear_transpose_p = core.Primitive("linear_transpose")
linear_transpose_p.def_impl(_linear_transpose_impl)


@linear_transpose_p.def_abstract_eval
def linear_transpose_p_abstract_eval(f, *, argnums):
  i_spec = f.arg_spec
  o_spec = f.ret_spec

  if argnums is None:
    i_spec_t = (o_spec,)
    o_spec_t = i_spec
    param_names = ["cotangent"]
  else:
    i_spec_t = list(i_spec)
    i_spec_t[argnums] = o_spec
    o_spec_t = i_spec[argnums]
    param_names = list(f"arg{i}" for i in range(len(i_spec_t)))
    param_names[argnums] = "cotangent"

  return GeneralArray(
    (Ret(o_spec_t), *(Arg(s, name=n) for s, n in zip(i_spec_t, param_names)))
  )


def _linear_transpose_transpose_rule(t, f, *, argnums):
  assert ad.is_undefined_primal(f)
  if argnums is None:
    tt = split(unpack_args(linear_transpose(t, argnums=argnums)))[0]
  else:
    tt = linear_transpose(t, argnums=argnums)
  if f.aval.shape != tt.shape:
    raise RuntimeError("Internal error")
  return (tt,)


jax.interpreters.ad.deflinear2(
  linear_transpose_p, _linear_transpose_transpose_rule
)
jax.interpreters.ad.primitive_transposes[linear_transpose_p
                                        ] = _linear_transpose_transpose_rule


def integrate(f):
  """Integrate function f over all input arguments.
  WARNING, the forward pass of this function is not implemented.
  This is only useful for computing the functional gradient of
  integral type functionals.

  Args:
    f: Function to integrate.
  Returns:
    Scalar representing the integral of f.
  """
  return integrate_p.bind(f)


def _integrate_impl(f):
  return dummy_array(f.ret_spec)


integrate_p = core.Primitive("integrate")
integrate_p.def_impl(_integrate_impl)


@integrate_p.def_abstract_eval
def integrate_p_abstract_eval(f):
  return tree_map(
    lambda spec: core.ShapedArray(spec.shape, dtype=spec.dtype),
    f.ret_spec,
  )


def _integrate_transpose_rule(t, f):
  assert ad.is_undefined_primal(f)
  # TODO: this needs to use primitive

  @with_spec(f.aval.arg_spec, f.aval.ret_spec)
  def return_t(*args, **kwargs):
    return t

  return (return_t,)


jax.interpreters.ad.deflinear2(integrate_p, _integrate_transpose_rule)
jax.interpreters.ad.primitive_transposes[integrate_p
                                        ] = _integrate_transpose_rule

# compose function f with gs


def compose(f, *gs, **params):
  """Compose function f with all the gs.

  Equivalent to:

  .. code-block:: python

    def compose(f, *gs):
      def composed(*args):
        return f(
          *(g(*arg) for arg, g in zip(args, gs))
        )
      return composed

  Args:
    f: function to compose with.
    *gs: functions to compose f with.
  Returns:
    A function that applies f to the outputs of gs.
  """
  return compose_p.bind(f, *gs, **params)


def _compose_impl(f, *gs, **params):
  """Compose the function f and function g, apply g first and then f.
  """
  _assert_composable(f, *gs)
  share_inputs = params.get("share_inputs", False)
  if share_inputs:
    _assert_same_input_signature(*gs)
  gs_arg_spec = tuple(g.arg_spec for g in gs)
  if len(gs) == 1 or share_inputs:
    combined_arg_spec = gs_arg_spec[0]
  else:
    combined_arg_spec = gs_arg_spec

  @with_spec(combined_arg_spec, f.ret_spec)
  def fgs(*args, **kwargs):
    if len(gs) > 1 and not share_inputs:
      f_args = tuple(g(*a) for g, a in zip(gs, args))
    else:
      f_args = tuple(g(*args, **kwargs) for g in gs)
    return f(*f_args)

  return fgs


compose_p = core.Primitive("compose")
compose_p.def_impl(_compose_impl)


@compose_p.def_abstract_eval
def compose_p_abstract_eval(f, *gs, **params):
  _assert_composable(f, *gs)
  share_inputs = params.get("share_inputs", False)
  if share_inputs:
    _assert_same_input_signature(*gs)
  if len(gs) > 1 and not share_inputs:
    arg_dims = (Arg(g.arg_spec, name=f"arg{i}") for i, g in enumerate(gs))
  else:
    arg_dims = gs[0].shape[1:]
  return GeneralArray((f.shape[0], *arg_dims))


def _compose_jvp_rule(primals, tangents, **params):
  share_inputs = params.get("share_inputs", False)
  f_type = params.get("f_type", "unknown")
  f, gs = primals[0], primals[1:]
  f_dot, gs_dot = tangents[0], tangents[1:]
  primals_out = compose(f, *gs, **params)
  tangents_out = []
  # contribution from f_dot
  if not isinstance(f_dot, ad_util.Zero):
    tangents_out.append(compose(f_dot, *gs))

  # contribution from gs_dot
  if f_type == "unknown":
    # linearize f
    tangents_out.append(
      compose(
        linearize(f),
        zip_functions(*gs, share_inputs=share_inputs),
        zip_functions(*gs_dot, share_inputs=share_inputs),
        share_inputs=True,
        f_type="linearized",
      )
    )
  elif f_type == "linear":
    tangents_out.append(compose(f, *gs_dot, **params))
  elif f_type == "linearized":
    raise NotImplementedError

  # one more step to sum contributions from f_dot and gs_dot
  return primals_out, add(*tangents_out)


def _compose_transpose_rule(cotangent, f, *gs, **params):
  share_inputs = params.get("share_inputs", False)
  f_type = params.get("f_type", "unknown")

  if isinstance(cotangent, ad_util.Zero):
    return (None, *(ad_util.Zero(a.aval) for a in gs))

  if ad.is_undefined_primal(f):
    # when gs are fixed, the compose rule is linear to f
    # because compose(f1, *gs) + compose(f2, *gs) = compose(f1 + f2, *gs)
    # cotangent: A -> C, gs: A -> B.
    # However, this requires us to know the inverse of gs,
    # so that we construct inverse g: B -> A with cotangent: A -> C into B -> C
    # We currently just disable the transpose rule wrt f.
    # TODO: maybe when gs are invertible, we could enable it.
    raise RuntimeError(
      "transpose of compose(f, g1, g2, ...) wrt f is not defined."
    )
  else:
    # when f is linear, the compose rule is linear to each gs
    # cotangent: A -> C, f: B -> C  => A -> B
    # we need to apply linear_transpose on function f elementwisely.
    # Then, we can compose(linear_transposed_f, cotangent),
    # which generates a function of A -> B.
    if f_type == "linear":
      wrt_gs = compose(
        linear_transpose(f, argnums=None), cotangent, f_type="linear"
      )
      return (None, *split(wrt_gs))
    elif f_type == "unknown":
      raise RuntimeError("Fail to transpose because f_type is not specified.")
    elif f_type == "linearized":
      if len(gs) != 2:
        raise RuntimeError(
          "_compose_transpose_rule is only applied to the "
          "jvp graph, which means the function f linearized "
          "to take exactly one primal and one tangent argument. "
        )
      if not share_inputs:
        raise RuntimeError(
          "The composed primal / tangent functions must also share inputs."
        )
      primal, tangent = gs
      if ad.is_undefined_primal(primal):
        raise RuntimeError(
          "transpose of compose(linearized_f, g1, g2) wrt g1 is not defined. "
          "Because it may not be a linear function."
        )
      elif ad.is_undefined_primal(tangent):
        wrt_tangent = compose(
          linear_transpose(f, argnums=1),
          primal,
          cotangent,
          f_type="linearized",
          share_inputs=True,
        )
        return (None, None, wrt_tangent)


jax.interpreters.ad.primitive_jvps[compose_p] = _compose_jvp_rule
jax.interpreters.ad.primitive_transposes[compose_p] = _compose_transpose_rule


def nabla(f, argnums=0, has_aux=False):
  """nabla operator obtains the gradient function.
  In the simplest case, we consider single variable single output function.
  f: R -> R, (nabla f): R -> R.
  While we can generalize to multiple variables and multiple outputs.
  for f: R^n -> R^m, (nabla f): R^n -> R^(m x n).

  Equivalent to:

  .. code-block:: python

    def nabla(f, argnums=0, has_aux=False):
      def nabla_f(*args):
        jax.jacfwd(f, argnums, has_aux=has_aux)(*args)
      return nabla_f

  Args:
    f: function to be differentiated.
    argnums: the argument index to be differentiated.
    has_aux: whether the function has auxilary outputs.
  Returns:
    (nabla f): the gradient function.
  """
  return nabla_p.bind(f, argnums=argnums, has_aux=has_aux)


def _nabla_impl(f, *, argnums=0, has_aux=False):
  jac_spec = jacobian_spec(f, argnums, has_aux)
  jacfwd = jax.jacfwd(f, argnums=argnums, has_aux=has_aux)

  @with_spec(f.arg_spec, jac_spec)
  def _jac_f(*args, **kwargs):
    return jacfwd(*args, **kwargs)

  return _jac_f


nabla_p = core.Primitive("nabla")
nabla_p.def_impl(_nabla_impl)


@nabla_p.def_abstract_eval
def nabla_p_abstract_eval(
  f,
  *,
  argnums=0,
  has_aux=False,
):
  # normalize argnums to tuple
  spec = jacobian_spec(f, argnums, has_aux)
  return GeneralArray((Ret(spec), *f.shape[1:]))


def _nabla_transpose_rule(t, f, *, argnums=0, has_aux=False):
  cotangent_spec = t.ret_spec
  assert ad.is_undefined_primal(f)
  jac_spec = jacobian_spec(f.aval, argnums, has_aux)
  if (cotangent_spec != jac_spec):
    raise ValueError(
      f"The cotangent is expected to be {jac_spec}, got {cotangent_spec}."
    )
  o_spec = f.aval.ret_spec
  nabla_t = nabla(t, argnums=argnums, has_aux=has_aux)

  @with_spec((nabla_t.ret_spec,), f.aval.ret_spec)
  def _contract(arg, /):
    """The transpose of nabla is simply -nabla when the operand is
    a scalar->scalar function. However, here we generalized to the
    case where both input/output of the function can be a pytree of
    arrays.
    """
    if has_aux:
      arg, aux = arg
    else:
      aux = None
    # negative nabla
    arg = tree_map(lambda y: -y, arg)
    out = tree_map(
      lambda spec, x:
      sum([xf.reshape(*spec.shape, -1).sum(-1) for xf in tree_flatten(x)[0]]),
      o_spec,
      arg,
    )
    return ((out, aux) if aux is not None else out)

  ret = compose(_contract, nabla_t)
  return (ret,)


def _nabla_jvp_rule(primals, tangents, *, argnums=0, has_aux=False):
  primal_out = nabla(primals[0], argnums=argnums, has_aux=has_aux)
  tangent_out = nabla(tangents[0], argnums=argnums, has_aux=has_aux)
  return primal_out, tangent_out


jax.interpreters.ad.primitive_jvps[nabla_p] = _nabla_jvp_rule
jax.interpreters.ad.primitive_transposes[nabla_p] = _nabla_transpose_rule


def linearize(f, *, argnums=None):
  """Linearize is a bit different from the jax.linearize call
  The reason is that jax.linearize requires both function and the
  primal values. As an operator, the linearize function here takes
  as input a function f. And returns a function lf(primal, tangent).

  Equivalent to:

  .. code-block:: python

    def linearize(f):
      def linearized_f(primal, tangent):
        return jax.jvp(f, primal, tangent)
      return linearized_f

  Args:
    f: function to be differentiated.
    argnums: the argument index to be differentiated.
      default to None, which computes linearization wrt all arguments.
  Returns:
    (linear f): the linearized function that takes primal and tangent.
  """
  argnums = _normalize_argnums(f, argnums)
  return linearize_p.bind(f, argnums=argnums)


def _linearize_impl(f, *, argnums=None):

  @with_spec(
    (f.arg_spec, tuple(f.arg_spec[idx] for idx in argnums)), f.ret_spec
  )
  def linearized_f(primals, tangents, /):

    def partial_f(*args):
      _primals = [p for p in primals]
      for i, idx in enumerate(argnums):
        _primals[idx] = args[i]
      return f(*_primals)

    primal_args = tuple(primals[idx] for idx in argnums)
    return jax.jvp(partial_f, primal_args, tangents)[1]

  return linearized_f


linearize_p = core.Primitive("linearize")
linearize_p.def_impl(_linearize_impl)


@linearize_p.def_abstract_eval
def linearize_p_abstract_eval(f, *, argnums=None):
  primals_spec = f.arg_spec
  tangents_spec = tuple(primals_spec[i] for i in argnums)
  ret_spec = f.shape[0].spec
  return GeneralArray(
    (
      Ret(ret_spec),
      Arg(primals_spec, name="primals"),
      Arg(tangents_spec, name="tangents"),
    )
  )


def _linearize_transpose_rule(t, f, *, argnums=None):
  # t: (primal, tangent) -> (primal_out, tangent_out)
  # f: (primal -> out)
  # seems we need to integrate over one of the inputs.
  # not very sure as this will go to infinity.
  raise NotImplementedError


def _linearize_jvp_rule(primals, tangents, *, argnums=None):
  # this is supposed to be an linear operator
  # because it consists of applying a nabla operator on the input function
  # and then product with an identity function.
  f_dot = tangents[0]
  return linearize(f_dot, argnums=argnums)


jax.interpreters.ad.primitive_jvps[linearize_p] = _linearize_jvp_rule
jax.interpreters.ad.primitive_transposes[linearize_p
                                        ] = _linearize_transpose_rule


def unpack_args(f):
  """Unpack the parameter of function f.
  The original function f takes a single input, which is a tuple.
  It converts f into a function that takes multiple inputs which
  are unpacked from the single input.

  Equivalent to:
  .. code-block:: python

    def unpack_args(f):
      def unpacked_f(*args):
        return f(args)
      return unpacked_f

  Args:
    f: input function.
  Returns:
    (unpacked f): the function that takes multiple arguments.
  """
  return unpack_args_p.bind(f)


def _unpack_args_impl(f):
  i_spec = f.arg_spec
  if len(i_spec) != 1 or not isinstance(i_spec[0], Tuple):
    raise ValueError("f must take a single tuple type argument.")

  @with_spec(i_spec[0], f.ret_spec)
  def _unpacked_f(*args):
    return f(args)

  return _unpacked_f


unpack_args_p = core.Primitive("unpack_args")
unpack_args_p.def_impl(_unpack_args_impl)


@unpack_args_p.def_abstract_eval
def unpack_args_p_abstract_eval(f):
  spec = f.arg_spec
  if len(spec) != 1 or not isinstance(spec[0], tuple):
    raise ValueError(
      "f must be a function that takes only one argument of Tuple type."
    )
  return GeneralArray(
    (f.shape[0], *(Arg(s, name=f"arg{i}") for i, s in enumerate(spec[0])))
  )


def _unpack_args_transpose_rule(t, f):
  return (pack_args(t),)


jax.interpreters.ad.deflinear2(unpack_args_p, _unpack_args_transpose_rule)
jax.interpreters.ad.primitive_transposes[unpack_args_p
                                        ] = _unpack_args_transpose_rule


def pack_args(f):
  """Pack the parameter of function f.
  The original function f takes multiple inputs.
  It converts f into a function that takes a single input which is
  a tuple packed from the original inputs.

  Equivalent to:
  .. code-block:: python

    def pack_args(f):
      def packed_f(args_tuple):
        return f(*args_tuple)
      return packed_f

  Args:
    f: input function.
  Returns:
    (packed f): the function that takes a single tuple.
  """
  return pack_args_p.bind(f)


def _pack_args_impl(f):

  @with_spec((f.arg_spec,), f.ret_spec)
  def _packed_f(args):
    return f(*args)

  return _packed_f


pack_args_p = core.Primitive("pack_args")
pack_args_p.def_impl(_pack_args_impl)


@pack_args_p.def_abstract_eval
def pack_args_p_abstract_eval(f):
  spec = f.arg_spec
  return GeneralArray((f.shape[0], Arg(spec, name="args")))


def _pack_args_transpose_rule(t, unused_f):
  return (unpack_args(t),)


jax.interpreters.ad.deflinear2(pack_args_p, _pack_args_transpose_rule)
jax.interpreters.ad.primitive_transposes[pack_args_p
                                        ] = _pack_args_transpose_rule


def _unary_compose(u, f):
  """Compose unary function `u` with the input function f.

  Args:
    f: the input function.
  Returns:
    out: `compose(u, f)`.
  """

  @with_spec((f.ret_spec,), f.ret_spec)
  def _u(x):
    return tree_map(u, x)

  return compose(_u, f)


unary_funcs = {
  "negative": jnp.negative,
  "exp": jnp.exp,
  "log": jnp.log,
  "sin": jnp.sin,
  "cos": jnp.cos,
  "tan": jnp.tan,
  "arcsin": jnp.arcsin,
  "arccos": jnp.arccos,
  "arctan": jnp.arctan,
  "sinh": jnp.sinh,
  "cosh": jnp.cosh,
  "tanh": jnp.tanh,
  "arcsinh": jnp.arcsinh,
  "arccosh": jnp.arccosh,
  "arctanh": jnp.arctanh,
  "square": jnp.square,
  "sqrt": jnp.sqrt,
  "abs": jnp.abs,
  "erf": jax.lax.erf,
  "sigmoid": jax.nn.sigmoid,
  "relu": jax.nn.relu,
}

for u_name, u in unary_funcs.items():
  u_compose = partial(_unary_compose, u)
  u_compose.__doc__ = _unary_compose.__doc__.format(u_name)
  globals()[u_name] = u_compose


def broadcast_functions(*fs):
  specs = []
  max_num_leaves = 0
  full_tree = None
  arg_spec = None
  # find the spec trees of each input
  # and keep the largest spec tree.
  for f in fs:
    if hasattr(f, "aval"):
      f = f.aval
    if isinstance(f, (function, GeneralArray)):
      spec = f.ret_spec
      if arg_spec is None:
        arg_spec = f.arg_spec
      else:
        if f.arg_spec != arg_spec:
          raise ValueError(
            f"Cannot broadcast functions with different input signatures. "
            f"Got {arg_spec} and {f.arg_spec}"
          )
    elif isinstance(f, core.ShapedArray):
      spec = SpecTree.from_value(f)
    else:
      spec = SpecTree.from_value(jnp.asarray(f))
    specs.append(spec)
    num_leaves = tree_structure(spec).num_leaves
    if num_leaves > max_num_leaves:
      max_num_leaves = num_leaves
      full_tree = spec

  if arg_spec is None:
    raise ValueError(f"None of {fs} is a function.")
  # now broadcast all the spec trees to the full tree
  specs = tuple(
    tree_unflatten(
      tree_structure(full_tree), broadcast_prefix(spec, full_tree)
    ) for spec in specs
  )
  # extract the shapes of the full tree
  full_shapes = tree_map(
    lambda *args: jnp.broadcast_shapes(*(a.shape for a in args)), *specs
  )

  # now create the spec with full tree and full shapes
  # while keeping the dtype
  specs = tuple(
    tree_map(lambda sp, sh: Spec(sh, sp.dtype), spec, full_shapes)
    for spec in specs
  )

  # compose this function with any of fs, they will be broadcasted to
  # full tree and full shapes.
  def _broadcast(spec, value):
    broadcast_tree = tree_unflatten(
      tree_structure(spec), broadcast_prefix(value, spec)
    )
    return tree_map(
      lambda v, s: jnp.broadcast_to(v, s.shape), broadcast_tree, spec
    )

  def _make_broadcast_fn(spec, f):
    bcast = function(partial(_broadcast, spec), (f.ret_spec,), spec)
    return compose(bcast, f)

  # for values, we make a constant function out of the value.
  def _make_constant_fn(spec, value):

    @with_spec(arg_spec, spec)
    def _constant(*args):
      return value

    return _constant

  # now we iterate all the functions
  # and transform each function to have the full signature.
  ret = []
  for f, spec in zip(fs, specs):
    f_aval = getattr(f, "aval", f)
    if isinstance(f_aval, (function, GeneralArray)):
      ret_spec = f_aval.ret_spec
      if ret_spec != spec:
        f = _make_broadcast_fn(spec, f)
      ret.append(f)
      continue
    elif isinstance(f_aval, core.ShapedArray):
      pass
    else:
      f = jnp.asarray(f)
    if SpecTree.from_value(f) != spec:
      f = _broadcast(spec, f)
    ret.append(_make_constant_fn(spec, f))
  return ret


def sub(x, y):
  x, y = broadcast_functions(x, y)
  dummy = tree_map(jnp.subtract, dummy_output(x), dummy_output(y))

  @with_spec((x.ret_spec, y.ret_spec), SpecTree.from_value(dummy))
  def _sub(x, y):
    return tree_map(lambda x, y: x - y, x, y)

  return compose(_sub, x, y, share_inputs=True, f_type="linear")


def add(*fs):
  """Add functions together.

  Equivalent to:

  .. code-block:: python

    def add(*fs):
      def summed(*args):
        return sum(f(*args) for f in fs)
      return summed

  Args:
    *fs: functions to add together.
  Returns:
    A function that sums the outputs of the functions.
  """
  fs = broadcast_functions(*fs)
  fs_o_spec = tuple(f.ret_spec for f in fs)

  @with_spec(arg_spec=fs_o_spec, ret_spec=fs_o_spec[0])
  def _add(*args):
    return tree_map(lambda *x: sum(x), *args)

  return compose(_add, *fs, share_inputs=True, f_type="linear")


ad_util.jaxval_adders[types.FunctionType] = add
ad_util.jaxval_adders[function] = add


def div(x, y):
  x, y = broadcast_functions(x, y)
  dummy = tree_map(jnp.divide, dummy_output(x), dummy_output(y))

  @with_spec((x.ret_spec, y.ret_spec), SpecTree.from_value(dummy))
  def _div(x, y):
    return tree_map(lambda x, y: x / y, x, y)

  return compose(_div, x, y, share_inputs=True)


def mul(x, y):
  x, y = broadcast_functions(x, y)
  dummy = tree_map(jnp.multiply, dummy_output(x), dummy_output(y))

  @with_spec((x.ret_spec, y.ret_spec), SpecTree.from_value(dummy))
  def _mul(x, y):
    return tree_map(lambda x, y: x * y, x, y)

  return compose(_mul, x, y, share_inputs=True)


def pow(x, y):
  x, y = broadcast_functions(x, y)
  dummy = tree_map(jnp.power, dummy_output(x), dummy_output(y))

  @with_spec((x.ret_spec, y.ret_spec), SpecTree.from_value(dummy))
  def _pow(x, y):
    return tree_map(lambda x, y: x**y, x, y)

  return compose(_pow, x, y, share_inputs=True)


array_operators = {
  "neg": lambda self: negative(self),  # noqa
  "add": add,
  "radd": add,
  "sub": sub,
  "rsub": lambda y, x: sub(x, y),
  "truediv": div,
  "rtruediv": lambda y, x: div(x, y),
  "mul": mul,
  "rmul": mul,
  "pow": pow,
  "rpow": lambda y, x: pow(x, y),
}


def _set_function_operators(cls, operators):
  for op_name, op in operators.items():
    setattr(cls, f"__{op_name}__", op)


def _set_general_array_operators(cls, operators):
  for op_name, op in operators.items():
    setattr(cls, f"_{op_name}", staticmethod(op))


_set_function_operators(function, array_operators)
_set_general_array_operators(GeneralArray, array_operators)
