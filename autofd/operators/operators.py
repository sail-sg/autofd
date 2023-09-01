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
import inspect
import numpy as np
from typing import Tuple
from makefun import with_signature
from functools import partial, update_wrapper

import jax
import jax.numpy as jnp
from jax.tree_util import (
  tree_map,
  tree_flatten,
)
import jax.interpreters.ad as ad
import jax._src.ad_util as ad_util
import jax._src.core as core
from jax._src import custom_api_util

from autofd.general_array import (
  Ret,
  Arg,
  SpecTree,
  GeneralArray,
  general_shape,
  signature,
  parameters,
  num_args,
  jacobian_spec,
  dummy_array,
  return_annotation,
  function_to_aval,
)


def _assert_same_input_signature(*fs):
  arg_specs = tuple(SpecTree.from_args(f) for f in fs)
  for d in arg_specs[1:]:
    if d != arg_specs[0]:
      raise ValueError("All functions must have the same input signature.")


def _assert_same_output_signature(*fs):
  arg_specs = tuple(SpecTree.from_ret(f) for f in fs)
  for d in arg_specs[1:]:
    if d != arg_specs[0]:
      raise ValueError("All functions must have the same output signature.")


def _assert_composable(f, *gs):
  f_i_spec = SpecTree.from_args(f)
  gs_o_spec = tuple(SpecTree.from_ret(g) for g in gs)
  if f_i_spec != gs_o_spec:
    raise ValueError(
      f"Cannot compose functions with incompatible shapes: "
      f"f has shape: {general_shape(f)}, "
      f"while g are {tuple(general_shape(g) for g in gs)}"
    )


def _normalize_argnums(f, argnums):
  num_args_f = num_args(f)
  if argnums is None:
    argnums = tuple(range(num_args_f))
  elif isinstance(argnums, int):
    argnums = (argnums,)
  if isinstance(argnums, tuple):
    if len(set(argnums)) < len(argnums):
      raise ValueError("argnums must be unique.")
    elif any(map(lambda x: x >= num_args_f, argnums)):
      raise ValueError("argnums must be less than the number of arguments.")
  return argnums


@custom_api_util.register_custom_decorator_type
class function:
  """Wrapper for a function, to add some operator overloading.
  TODO: allow other to be scalar
  """

  def __init__(self, f):
    update_wrapper(self, f)
    self.f = f
    self.ret_ann = return_annotation(self.f)

  def __call__(self, *args):
    return self.f(*args)

  def __radd__(self, other):
    return self.__add__(other)

  def __add__(self, other):
    _assert_same_input_signature(self.f, other)
    _assert_same_output_signature(self.f, other)

    def add(x: self.ret_ann, y: self.ret_ann) -> self.ret_ann:
      return tree_map(jnp.add, x, y)

    return compose(add, self.f, other, share_inputs=True)

  def __neg__(self):

    def neg(x: self.ret_ann) -> self.ret_ann:
      return tree_map(jnp.negative, x)

    return compose(neg, self.f)

  def __rsub__(self, other):
    _assert_same_input_signature(self.f, other)
    _assert_same_output_signature(self.f, other)

    def sub(x: self.ret_ann, y: self.ret_ann) -> self.ret_ann:
      return tree_map(jnp.subtract, x, y)

    return compose(sub, other, self.f, share_inputs=True)

  def __sub__(self, other):
    _assert_same_input_signature(self.f, other)
    _assert_same_output_signature(self.f, other)

    def sub(x: self.ret_ann, y: self.ret_ann) -> self.ret_ann:
      return tree_map(jnp.subtract, x, y)

    return compose(sub, self.f, other, share_inputs=True)

  def __rmul__(self, other):
    return self.__mul__(other)

  def __mul__(self, other):
    _assert_same_input_signature(self.f, other)
    _assert_same_output_signature(self.f, other)

    def mul(x: self.ret_ann, y: self.ret_ann) -> self.ret_ann:
      return tree_map(jnp.multiply, x, y)

    return compose(mul, self.f, other, share_inputs=True)

  def __rtruediv__(self, other):
    _assert_same_input_signature(self.f, other)
    _assert_same_output_signature(self.f, other)

    def div(x: self.ret_ann, y: self.ret_ann) -> self.ret_ann:
      return tree_map(jnp.divide, x, y)

    return compose(div, other, self.f, share_inputs=True)

  def __truediv__(self, other):
    _assert_same_input_signature(self.f, other)
    _assert_same_output_signature(self.f, other)

    def div(x: self.ret_ann, y: self.ret_ann) -> self.ret_ann:
      return tree_map(jnp.divide, x, y)

    return compose(div, self.f, other, share_inputs=True)

  def __pow__(self, exponent):

    def _pow(x: self.ret_ann) -> self.ret_ann:
      return tree_map(lambda x: jnp.power(x, exponent), x)

    return compose(_pow, self.f)


jax.core.pytype_aval_mappings[function] = function_to_aval
jax.interpreters.xla.pytype_aval_mappings[function] = function_to_aval
jax._src.api_util._shaped_abstractify_handlers[function] = function_to_aval
jax._src.dtypes.python_scalar_dtypes[function] = np.dtype("float32")


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
  fs_o_spec = tuple(SpecTree.from_ret(f) for f in fs)

  @with_signature(
    inspect.Signature(
      (
        SpecTree.to_parameter(SpecTree.from_ret(f), name=f"arg{i}")
        for i, f in enumerate(fs)
      ),
      return_annotation=SpecTree.to_annotation(fs_o_spec),
    )
  )
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
  o_spec = SpecTree.from_ret(f)
  if not isinstance(o_spec, tuple):
    raise ValueError("Function must return a tuple.")

  def _split_i(i, *args, **kwargs):
    return f(*args, **kwargs)[i]

  fns = []
  for i, spec in enumerate(o_spec):
    fns.append(
      function(
        with_signature(
          inspect.Signature(
            parameters(f),
            return_annotation=SpecTree.to_annotation(spec),
          )
        )(partial(_split_i, i))
      )
    )
  return tuple(fns)


split_p = core.Primitive("split")
split_p.def_impl(_split_impl)
split_p.multiple_results = True


@split_p.def_abstract_eval
def split_p_abstract_eval(f):
  ret_spec = SpecTree.from_ret(f)
  if not isinstance(ret_spec, tuple):
    raise ValueError("function must return a tuple.")
  return tuple(GeneralArray((Ret(s), *f.shape[1:])) for s in ret_spec)


def _split_transpose_rule(t, f):
  return (concat(*t),)


jax.interpreters.ad.deflinear2(split_p, _split_transpose_rule)
jax.interpreters.ad.primitive_transposes[split_p] = _split_transpose_rule


def zip_functions(*fs, share_inputs=False):
  zip_params = tuple(
    inspect.Parameter(
      name=f"x{i}",
      kind=inspect.Parameter.POSITIONAL_ONLY,
      annotation=return_annotation(f)
    ) for i, f in enumerate(fs)
  )
  zip_returns = Tuple[*(p.annotation for p in zip_params)]

  @with_signature(inspect.Signature(zip_params, return_annotation=zip_returns))
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
  i_spec = SpecTree.from_args(f)
  o_spec = SpecTree.from_ret(f)

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

  sig = inspect.Signature(
    tuple(
      SpecTree.to_parameter(s, name=n) for s, n in zip(i_spec_t, param_names)
    ),
    return_annotation=SpecTree.to_annotation(o_spec_t),
  )

  @function
  @with_signature(sig)
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
  i_spec = SpecTree.from_args(f)
  o_spec = SpecTree.from_ret(f)

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
  if argnums is None:
    tt = split(unpack_args(linear_transpose(t, argnums=argnums)))[0]
  else:
    tt = linear_transpose(t, argnums=argnums)
  if general_shape(f) != general_shape(tt):
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
  return dummy_array(SpecTree.from_ret(f))


integrate_p = core.Primitive("integrate")
integrate_p.def_impl(_integrate_impl)


@integrate_p.def_abstract_eval
def integrate_p_abstract_eval(f):
  spec_tree = SpecTree.from_ret(f)
  return tree_map(
    lambda spec: core.ShapedArray(spec.shape, dtype=spec.dtype),
    spec_tree,
  )


def _integrate_transpose_rule(t, f):

  # TODO: this needs to use primitive
  @function
  @with_signature(signature(f))
  def return_t(*args, **kwargs):
    return t

  return (return_t,)


jax.interpreters.ad.deflinear2(integrate_p, _integrate_transpose_rule)
jax.interpreters.ad.primitive_transposes[integrate_p
                                        ] = _integrate_transpose_rule


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
  fs_o_spec = tuple(SpecTree.from_ret(f) for f in fs)

  @function
  @with_signature(
    inspect.Signature(
      (
        SpecTree.to_parameter(spec, name=f"arg{i}")
        for i, spec in enumerate(fs_o_spec)
      ),
      return_annotation=SpecTree.to_annotation(fs_o_spec[0]),
    )
  )
  def sum_output(*args):
    return tree_map(lambda *x: sum(x), *args)

  return compose(sum_output, *fs, share_inputs=True, f_type="linear")


ad_util.jaxval_adders[types.FunctionType] = add
ad_util.jaxval_adders[function] = add
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
  f_sig = signature(f)
  gs_parameters = tuple(parameters(g) for g in gs)
  if len(gs) == 1 or share_inputs:
    combined_params = gs_parameters[0]
  else:
    combined_params = tuple(
      inspect.Parameter(
        f"args{i}",
        kind=inspect.Parameter.POSITIONAL_ONLY,
        annotation=Tuple[*(p.annotation for p in g_parameters)]
      ) for i, g_parameters in enumerate(gs_parameters)
    )

  @function
  @with_signature(
    inspect.Signature(
      combined_params, return_annotation=f_sig.return_annotation
    )
  )
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
    arg_dims = (
      Arg(SpecTree.from_args(g), name=f"arg{i}") for i, g in enumerate(gs)
    )
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
        split(linearize(f))[1],
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
          "transpose of compose(f, g1, g2) wrt g1 is not defined. "
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

  @function
  @with_signature(
    inspect.Signature(
      parameters=parameters(f),
      return_annotation=SpecTree.to_annotation(jac_spec),
    )
  )
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
  cotangent_spec = SpecTree.from_ret(t)
  jac_spec = jacobian_spec(f, argnums, has_aux)
  if (cotangent_spec != jac_spec):
    raise ValueError(
      f"The cotangent is expected to be {jac_spec}, got {cotangent_spec}."
    )
  o_spec = SpecTree.from_ret(f)
  nabla_t = nabla(t, argnums=argnums, has_aux=has_aux)

  def _contract(arg: return_annotation(nabla_t), /) -> return_annotation(f):
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
  in_anns = tuple(SpecTree.to_annotation(s) for s in SpecTree.from_args(f))
  primals_ann = Tuple[*in_anns]
  tangents_ann = Tuple[*(in_anns[idx] for idx in argnums)]
  ret_ann = return_annotation(f)

  @function
  def linearized_f(primals: primals_ann, tangents: tangents_ann,
                   /) -> Tuple[ret_ann, ret_ann]:

    def partial_f(*args):
      _primals = [p for p in primals]
      for i, idx in enumerate(argnums):
        _primals[idx] = args[i]
      return f(*_primals)

    primal_args = tuple(primals[idx] for idx in argnums)
    return jax.jvp(partial_f, primal_args, tangents)

  return linearized_f


linearize_p = core.Primitive("linearize")
linearize_p.def_impl(_linearize_impl)


@linearize_p.def_abstract_eval
def linearize_p_abstract_eval(f, *, argnums=None):
  primals_spec = SpecTree.from_args(f)
  tangents_spec = tuple(primals_spec[i] for i in argnums)
  ret_spec = f.shape[0].spec
  return GeneralArray(
    (
      Ret((ret_spec, ret_spec)),
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
  i_spec = SpecTree.from_args(f)
  if len(i_spec) != 1 or not isinstance(i_spec[0], Tuple):
    raise ValueError("f must take a single tuple type argument.")

  @function
  @with_signature(
    inspect.Signature(
      (
        SpecTree.to_parameter(spec, name=f"arg{i}")
        for i, spec in enumerate(i_spec[0])
      ),
      return_annotation=return_annotation(f)
    )
  )
  def _unpacked_f(*args):
    return f(args)

  return _unpacked_f


unpack_args_p = core.Primitive("unpack_args")
unpack_args_p.def_impl(_unpack_args_impl)


@unpack_args_p.def_abstract_eval
def unpack_args_p_abstract_eval(f):
  spec = SpecTree.from_args(f)
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

  @function
  def _packed_f(args: SpecTree.to_annotation(SpecTree.from_args(f)
                                            )) -> return_annotation(f):
    return f(*args)

  return _packed_f


pack_args_p = core.Primitive("pack_args")
pack_args_p.def_impl(_pack_args_impl)


@pack_args_p.def_abstract_eval
def pack_args_p_abstract_eval(f):
  spec = SpecTree.from_args(f)
  return GeneralArray((f.shape[0], Arg(spec, name="args")))


def _pack_args_transpose_rule(t, unused_f):
  return (unpack_args(t),)


jax.interpreters.ad.deflinear2(pack_args_p, _pack_args_transpose_rule)
jax.interpreters.ad.primitive_transposes[pack_args_p
                                        ] = _pack_args_transpose_rule
