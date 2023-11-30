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
import numpy as np
import functools
from types import SimpleNamespace
import jax
import jax.extend.linear_util as lu
import jax.numpy as jnp
from jax.tree_util import (
  tree_map,
  tree_leaves,
  tree_structure,
)
from jax.interpreters import ad
from jax._src import ad_util
from jax import core
from jax._src.util import HashableFunction
from absl import logging
from autofd.utils import complement_argnums
from . import transforms as t
from autofd.general_array import (
  ret_like,
  Ret,
  Arg,
  Spec,
  SpecTree,
  GeneralArray,
  function,
  operator,
  with_spec,
  num_args,
  jacobian_spec,
  dummy_array,
  is_function,
  Grid,
)


def def_operator(prim, impl, spec, grid, multiple_results=False):
  prim.multiple_results = multiple_results

  def _set_shape_and_dtype(v, shape, dtype):
    if isinstance(v, function):
      v.shape = shape

  def _set_grid(v, grid):
    if isinstance(v, function):
      v.grid = grid

  @prim.def_impl
  def _impl(*args, **kwargs):
    f = impl(*args, **kwargs)
    shape_and_dtype = spec(*args, **kwargs)
    g = grid(*args, **kwargs)
    if multiple_results:
      for fi, (si, di), gi in zip(f, shape_and_dtype, g):
        _set_shape_and_dtype(fi, si, di)
        _set_grid(fi, gi)
    else:
      _set_shape_and_dtype(f, *shape_and_dtype)
      _set_grid(f, g)
    return f

  @prim.def_abstract_eval
  def _abstract_eval(*args, **kwargs):
    shape_and_dtype = spec(*args, **kwargs)
    if multiple_results:
      return tuple(
        GeneralArray(s) if d is None else core.ShapedArray(s, dtype=d)
        for s, d in shape_and_dtype
      )
    else:
      s, d = shape_and_dtype
      return (
        GeneralArray(s)
        if shape_and_dtype[1] is None else core.ShapedArray(s, dtype=d)
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
      f"f requires: {SpecTree.from_args(f)}, "
      f"while gs provides {tuple(SpecTree.from_ret(g) for g in gs)}"
    )


@operator
@lu.wrap_init
def partial_impl(f, *, args, argnums):
  assert len(argnums) == len(args)

  @function
  @lu.wrap_init
  def partial_f(*rest_args):
    all_args = [None] * f.num_args
    a, b = 0, 0
    for i in range(f.num_args):
      if a < len(argnums) and i == argnums[a]:
        all_args[i] = args[a]
        a += 1
      else:
        all_args[i] = rest_args[b]
        b += 1
    return f(*all_args)

  return partial_f


# permute_args


def permute_args(f, *, perm):
  return permute_args_p.bind(f, perm=tuple(perm))


@operator
@lu.wrap_init
def permute_args_impl(f, *, perm):
  return function(t.permute_args(f, perm))


def permute_args_spec(f, *, perm):
  return (f.ret, *(f.arg[i] for i in perm)), None


def permute_args_grid(f, *, perm):
  return f.grid.take(perm)


permute_args_p = core.Primitive("permute_args")
def_operator(
  permute_args_p,
  permute_args_impl,
  permute_args_spec,
  permute_args_grid,
)


def _permute_args_transpose_rule(t, f, *, perm):
  assert ad.is_undefined_primal(f)
  inv_perm = np.argsort(perm)
  return (permute_args_p.bind(t, perm=tuple(inv_perm.tolist())),)


jax.interpreters.ad.deflinear2(permute_args_p, _permute_args_transpose_rule)
jax.interpreters.ad.primitive_transposes[permute_args_p
                                        ] = _permute_args_transpose_rule

# split


def split(f):
  return split_p.bind(f)


@operator
@lu.wrap_init
def split_impl(f):
  return tuple(function(t.return_at(f, i)) for i in range(len(f.ret.spec)))


def split_spec(f):
  assert isinstance(f.ret.spec, tuple)
  return tuple(((r, *f.arg), None) for r in map(Ret, f.ret.spec))


def split_grid(f):
  return tuple(f.grid for _ in f.ret.spec)


split_p = core.Primitive("split")
def_operator(
  split_p,
  split_impl,
  split_spec,
  split_grid,
  multiple_results=True,
)


def _split_transpose_rule(t, f):
  return (zip_functions(*t, share_inputs=True),)


jax.interpreters.ad.deflinear2(split_p, _split_transpose_rule)
jax.interpreters.ad.primitive_transposes[split_p] = _split_transpose_rule

# zip_functions


def zip_functions(*fs, share_inputs=False):
  """Zip functions.
  This requires all `*fs` to have the same input signature.
  Returns a function that returns a tuple of the outputs of the functions.

  Equivalent to:

  .. code-block:: python

    def zip_functions(*fs):
      def zipped_fs(*args):
        return tuple(f(*args) for f in fs)
      return zipped_fs

  Args:
    *fs: functions to be zipped.
  Returns:
    A function that returns a tuple of the outputs of the functions.
  """
  return compose(
    HashableFunction(lambda *args: args, None),
    *fs,
    share_inputs=share_inputs,
  )


# broadcast_to


def broadcast_arg(f, target, *, argnums):
  argnums = (argnums,) if isinstance(argnums, int) else argnums
  return broadcast_arg_p.bind(f, target, argnums=argnums)


def broadcast_arg_impl(f, target, *, argnums):
  return function(t.broadcast_arg(f, target.shape, argnums))


def broadcast_arg_spec(f, target, *, argnums):
  assert f.ret == target.shape[0]
  return target.shape, None


def broadcast_arg_grid(f, target, *, argnums):
  c_argnums = complement_argnums(argnums, target.num_args)
  if hasattr(target, "grid"):
    grid = f.grid + target.grid.take(c_argnums)
  else:
    grid = f.grid + Grid.none(len(c_argnums))
  perm = np.argsort(argnums + c_argnums)
  return grid.take(perm)


broadcast_arg_p = core.Primitive("broadcast_arg")

def_operator(
  broadcast_arg_p,
  broadcast_arg_impl,
  broadcast_arg_spec,
  broadcast_arg_grid,
)


def _broadcast_arg_jvp_rule(primals, tangents, *, argnums):
  f, target = primals
  f_dot, target_dot = tangents
  primals_out = broadcast_arg_p.bind(f, target, argnums=argnums)
  if not isinstance(f_dot, ad_util.Zero):
    tangent_out = broadcast_arg_p.bind(f_dot, target, argnums=argnums)
  else:
    tangent_out = ad_util.Zero.from_value(target_dot)
  return primals_out, tangent_out


def _broadcast_arg_transpose_rule(t, f, target, *, argnums):
  c_argnums = complement_argnums(argnums, target.num_args)
  return (integrate(t, argnums=c_argnums), None)


jax.interpreters.ad.primitive_jvps[broadcast_arg_p] = _broadcast_arg_jvp_rule
jax.interpreters.ad.primitive_transposes[broadcast_arg_p
                                        ] = _broadcast_arg_transpose_rule

# linear_transpose


def linear_transpose(f, *, argnums=0, unpack=False):
  """Returns the transpose of a linear function."""
  return linear_transpose_p.bind(
    f,
    argnums=argnums,
    unpack=unpack,
  )


@operator
@lu.wrap_init
def linear_transpose_impl(f, *, argnums, unpack):
  assert isinstance(f, function)
  argnums_tuple = (argnums,) if isinstance(argnums, int) else argnums
  bound_argnums = complement_argnums(argnums, f.num_args)
  dummy_spec = tuple(map(lambda idx: f.arg_spec[idx], argnums_tuple))
  dummy_primals = dummy_array(dummy_spec)

  @function
  @lu.wrap_init
  def transposed_f(*args):
    if not unpack:
      bound_args, cotangent = args[:-1], args[-1]
    else:
      bound_args, cotangent = args[:-len(f.ret_spec)], args[-len(f.ret_spec):]
    partial_f = partial_impl(f, args=bound_args, argnums=bound_argnums)
    ret = jax.linear_transpose(partial_f.f.call_wrapped, *dummy_primals)(
      cotangent
    )
    return ret[0] if isinstance(argnums, int) else tuple(ret)

  return transposed_f


def linear_transpose_spec(f, *, argnums, unpack):
  c_argnums = complement_argnums(argnums, f.num_args)
  if isinstance(argnums, int):
    ret = f.arg[argnums]
  else:
    ret = Ret(tuple(f.arg[i].spec for i in argnums))
  if unpack:
    assert isinstance(f.ret.spec, tuple)
    return (
      ret, *map(f.arg.__getitem__, c_argnums), *map(Arg, f.ret.spec)
    ), None
  else:
    return (ret, *map(f.arg.__getitem__, c_argnums), f.ret), None


def linear_transpose_grid(f, *, argnums, unpack):
  c_argnums = complement_argnums(argnums, f.num_args)
  if unpack:
    return f.grid.take(c_argnums) + Grid.none(len(f.ret.spec))
  else:
    return f.grid.take(c_argnums) + Grid.none(1)


linear_transpose_p = core.Primitive("linear_transpose")
def_operator(
  linear_transpose_p,
  linear_transpose_impl,
  linear_transpose_spec,
  linear_transpose_grid,
)


def _linear_transpose_transpose_rule(t, f, *, argnums, unpack):
  assert ad.is_undefined_primal(f)
  o_spec = SpecTree.from_ret(f)
  argnums_tuple = (argnums,) if isinstance(argnums, int) else argnums
  rest_index = complement_argnums(argnums, num_args(f))
  # if the primal function unpacks its output into the args
  # of the cotangent function
  if not unpack:
    argnums_t = num_args(t) - 1
  else:
    n = num_args(t)
    argnums_t = tuple(range(n - len(o_spec), n))
  unpack_t = (not isinstance(argnums, int))
  ret = linear_transpose_p.bind(
    t,
    argnums=argnums_t,
    unpack=unpack_t,  # unpack if argnums is a tuple
  )
  perm = np.argsort(rest_index + argnums_tuple)
  ret = permute_args_p.bind(ret, perm=tuple(perm))
  return (ret,)


jax.interpreters.ad.deflinear2(
  linear_transpose_p, _linear_transpose_transpose_rule
)
jax.interpreters.ad.primitive_transposes[linear_transpose_p
                                        ] = _linear_transpose_transpose_rule

# integrate


def integrate(f, *, argnums=0):
  """Integrate function f over all input arguments.
  WARNING, the forward pass of this function is not implemented.
  This is only useful for computing the functional gradient of
  integral type functionals.

  Args:
    f: Function to integrate.
  Returns:
    Scalar representing the integral of f.
  """
  return integrate_p.bind(f, argnums=argnums)


@operator
@lu.wrap_init
def integrate_impl(f, *, argnums):
  if num_args(f) == 0:
    return f()
  argnums = (argnums,) if isinstance(argnums, int) else argnums
  c_argnums = complement_argnums(argnums, f.num_args)
  grid = f.grid.take(argnums)

  @lu.wrap_init
  def partially_integrated_f(*args):
    nargs = num_args(f)
    vf = f
    for argidx in argnums:
      in_axes = tuple(0 if i == argidx else None for i in range(nargs))
      vf = jax.vmap(vf, in_axes=in_axes, out_axes=0)
    idx = np.argsort(c_argnums + argnums)
    concat_args = (args + grid.nodes)
    full_args = [concat_args[i] for i in idx]
    out = vf(*full_args)
    for w in grid.weights:
      out = w @ out
    return out

  if any(n is None for n in grid.nodes):
    if len(c_argnums) == 0:
      logging.warning(
        "grids_and_weights is None, this will create a dummy integration "
        "that always returns 0. The gradient of this dummy integration would "
        "still work."
      )
      return tree_map(lambda s: jnp.zeros(s.shape, s.dtype), f.ret.spec)
    else:
      raise ValueError("grids_and_weights needs to be specified.")
  else:
    if len(c_argnums) == 0:
      return partially_integrated_f.call_wrapped()
    else:
      return function(partially_integrated_f)


def integrate_spec(f, *, argnums):
  c_argnums = complement_argnums(argnums, f.num_args)
  rest_args = tuple(map(f.arg.__getitem__, c_argnums))
  if len(rest_args) == 0:
    return f.ret.spec.shape, f.ret.spec.dtype
  else:
    return (f.ret, *rest_args), None


def integrate_grid(f, *, argnums):
  c_argnums = complement_argnums(argnums, f.num_args)
  return f.grid.take(c_argnums)


integrate_p = core.Primitive("integrate")
def_operator(
  integrate_p,
  integrate_impl,
  integrate_spec,
  integrate_grid,
)


def _integrate_jvp_rule(primals, tangents, *, argnums):
  f = primals[0]
  f_dot = tangents[0]
  primals_out = integrate_p.bind(f, argnums=argnums)
  if not isinstance(f_dot, ad_util.Zero):
    tangents_out = integrate_p.bind(f_dot, argnums=argnums)
  else:
    tangents_out = ad_util.Zero.from_value(primals_out)
  return primals_out, tangents_out


def _integrate_transpose_rule(t, f, *, argnums):
  assert ad.is_undefined_primal(f)
  if is_function(t):
    cargnums = complement_argnums(argnums, num_args(f))
    return (broadcast_arg(t, f.aval, argnums=cargnums),)
  else:

    @with_spec(SpecTree.from_args(f), SpecTree.from_ret(f))
    def return_t(*args):
      return t

    return (return_t,)


jax.interpreters.ad.primitive_jvps[integrate_p] = _integrate_jvp_rule
jax.interpreters.ad.primitive_transposes[integrate_p
                                        ] = _integrate_transpose_rule

# compose


def compose(f, *gs, share_inputs=False):
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
  if not isinstance(f, function) and not isinstance(f, jax.core.Tracer):
    if not callable(f):
      raise ValueError(f"f must be a function. Got: {f}")
    arg_spec = tuple(SpecTree.from_ret(g) for g in gs)
    try:
      ret_spec = SpecTree.from_value(jax.eval_shape(f, *arg_spec))
    except Exception as e:
      raise ValueError(
        "Cannot compose functions because applying "
        f"{f} on {', '.join(map(str, arg_spec))} fails."
      ) from e
    f = with_spec(arg_spec=arg_spec, ret_spec=ret_spec)(f)
  return compose_p.bind(f, *gs, share_inputs=share_inputs)


def compose_impl(f, *gs, share_inputs):
  return function(t.compose(f, gs, tuple(g.num_args for g in gs), share_inputs))


def compose_spec(f, *gs, share_inputs):
  _assert_composable(f, *gs)
  if share_inputs:
    _assert_same_input_signature(*gs)
    return (f.ret, *gs[0].arg), None
  return (f.ret, *(a for g in gs for a in g.arg)), None


def compose_grid(f, *gs, share_inputs):
  if share_inputs:
    return gs[0].grid
  else:
    return sum(g.grid for g in gs)


compose_p = core.Primitive("compose")
def_operator(
  compose_p,
  compose_impl,
  compose_spec,
  compose_grid,
)


def _compose_jvp_rule(primals, tangents, *, share_inputs):
  f, gs = primals[0], primals[1:]
  f_dot, gs_dot = tangents[0], tangents[1:]
  primals_out = compose_p.bind(f, *gs, share_inputs=share_inputs)
  tangents_out = []
  # contribution from f_dot
  if not isinstance(f_dot, ad_util.Zero):
    tangents_out.append(compose_p.bind(f_dot, *gs, share_inputs=share_inputs))

  # contribution from gs_dot
  nonzero_gs_dot = []
  nonzero_indice = []
  argnums = []
  argidx = 0
  for i, g_dot in enumerate(gs_dot):
    if not isinstance(g_dot, ad_util.Zero):
      nonzero_gs_dot.append(g_dot)
      nonzero_indice.append(i)
      argnums.extend(list(range(argidx, argidx + num_args(g_dot))))
    argidx += num_args(g_dot)
  argnums = tuple(argnums)

  gs = zip_functions(*gs, share_inputs=share_inputs)
  gs_dot = zip_functions(*nonzero_gs_dot, share_inputs=share_inputs)
  if not share_inputs:
    gs_dot = broadcast_arg(gs_dot, gs, argnums=argnums)

  tangents_out.append(
    compose_p.bind(
      linearize_p.bind(f, argnums=tuple(nonzero_indice)),
      *split_p.bind(gs),
      *split_p.bind(gs_dot),
      share_inputs=True,
    )
  )

  # one more step to sum contributions from f_dot and gs_dot
  return primals_out, add(*tangents_out)


def _compose_transpose_rule(cotangent, f, *gs, share_inputs):

  if isinstance(cotangent, ad_util.Zero):
    return (None, *(ad_util.Zero.from_value(a) for a in gs))

  if ad.is_undefined_primal(f):
    # when gs are fixed, the compose rule is linear to f
    # because compose(f1, *gs) + compose(f2, *gs) = compose(f1 + f2, *gs)
    # cotangent: A -> C, gs: A -> B.
    # However, this requires us to know the inverse of gs,
    # so that we construct inverse g: B -> A with cotangent: A -> C into B -> C
    # We currently just disable the transpose rule wrt f.
    # TODO: maybe when gs are invertible, we could enable it.
    raise NotImplementedError(
      "transpose of compose(f, g1, g2, ...) wrt f is not defined."
    )
  else:
    # when f is linear, the compose rule is linear to each gs
    # cotangent: A -> C, f: B -> C  => A -> B
    # we need to apply linear_transpose on function f elementwisely.
    # Then, we can compose(linear_transposed_f, cotangent),
    # which generates a function of A -> B.
    # This could fail if f is not linear.
    narg = num_args(f)
    ret = [None] * (1 + narg)
    argnums, rest_args = [], []
    for i, g in enumerate(gs):
      if ad.is_undefined_primal(g):
        argnums.append(i)
      else:
        rest_args.append(g)
        ret[1 + i] = None
    argnums = tuple(argnums)
    if share_inputs or len(gs) == 1:
      wrt_tangent = compose_p.bind(
        linear_transpose(f, argnums=argnums),
        *rest_args,
        cotangent,
        share_inputs=share_inputs,
      )
    else:
      # TODO: implementing this requires a general method of
      # performing intergration, keep for future.
      # we first broadcast the args of the rest_args to the same as
      # the cotangent, and then compose the transposed f in the same way
      # as in the above share_inputs=True branch.
      # finally, we integrate over the args of the rest_args.
      raise NotImplementedError
    for i, tgt in zip(argnums, split_p.bind(wrt_tangent)):
      ret[1 + i] = tgt
    return tuple(ret)


jax.interpreters.ad.primitive_jvps[compose_p] = _compose_jvp_rule
jax.interpreters.ad.primitive_transposes[compose_p] = _compose_transpose_rule

# nabla


def nabla(f, *, argnums=0, method=jax.jacfwd):
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
  return nabla_p.bind(f, argnums=argnums, method=method)


@operator
@lu.wrap_init
def nabla_impl(f, *, argnums, method):
  return function(lu.wrap_init(method(f, argnums)))


def nabla_spec(f, *, argnums, method):
  ispec = f.arg[argnums].spec if isinstance(argnums, int) else tuple(
    f.arg[i].spec for i in argnums
  )
  ospec = f.ret.spec
  jspec = tree_map(
    lambda osp:
    tree_map(lambda isp: Spec(osp.shape + isp.shape, isp.dtype), ispec), ospec
  )
  return (Ret(jspec), *f.arg), None


def nabla_grid(f, *, argnums, method):
  return f.grid


nabla_p = core.Primitive("nabla")
def_operator(
  nabla_p,
  nabla_impl,
  nabla_spec,
  nabla_grid,
)


def _nabla_transpose_rule(t, f, *, argnums, method):
  cotangent_spec = SpecTree.from_ret(t)
  assert ad.is_undefined_primal(f)
  jac_spec = jacobian_spec(f.aval, argnums)
  if (cotangent_spec != jac_spec):
    raise ValueError(
      f"The cotangent is expected to be {jac_spec}, got {cotangent_spec}."
    )
  o_spec = SpecTree.from_ret(f)
  nabla_t = nabla(t, argnums=argnums, method=method)

  @with_spec((SpecTree.from_ret(nabla_t),), SpecTree.from_ret(f))
  def _contract(arg, /):
    """The transpose of nabla is simply -nabla when the operand is
    a scalar->scalar function. However, here we generalized to the
    case where both input/output of the function can be a pytree of
    arrays.
    """

    # negative nabla
    # arg = tree_map(lambda y: -y, arg)

    def neg_sum_diag(spec_prefix, xf):
      xf = tree_leaves(xf)
      ll = int(np.sqrt(len(xf)))
      out = []
      for i in range(ll):
        x = xf[i + i * ll]
        assert x.shape[:len(spec_prefix.shape)] == spec_prefix.shape
        l = int(np.sqrt(np.prod(x.shape[len(spec_prefix.shape):])))
        out.append(
          jnp.trace(x.reshape(spec_prefix.shape + (l, l)), axis1=-2, axis2=-1)
        )
      return -sum(out)

    out = tree_map(neg_sum_diag, o_spec, arg)
    return out

  ret = compose(_contract, nabla_t)
  return (ret,)


def _nabla_jvp_rule(primals, tangents, *, argnums, method):
  primal_out = nabla(primals[0], argnums=argnums, method=method)
  tangent_out = nabla(tangents[0], argnums=argnums, method=method)
  return primal_out, tangent_out


jax.interpreters.ad.primitive_jvps[nabla_p] = _nabla_jvp_rule
jax.interpreters.ad.primitive_transposes[nabla_p] = _nabla_transpose_rule

# linearize


def linearize(f, *, argnums=0):
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
  argnums = (argnums,) if isinstance(argnums, int) else argnums
  return linearize_p.bind(f, argnums=argnums)


@operator
@lu.wrap_init
def linearize_impl(f, argnums):
  assert isinstance(f, function)

  @function
  @lu.wrap_init
  def linearized_f(*args):
    assert len(args) == f.num_args + len(argnums)
    primals, tangents = args[:f.num_args], args[f.num_args:]
    bound_argnums = complement_argnums(argnums, f.num_args)
    unbound_args = tuple(map(primals.__getitem__, argnums))
    bound_args = tuple(map(primals.__getitem__, bound_argnums))
    partial_f = partial_impl(f, args=bound_args, argnums=bound_argnums)
    return jax.linearize(partial_f.f.call_wrapped, *unbound_args)[1](*tangents)

  return linearized_f


def linearize_spec(f, argnums):
  return (f.ret, *f.arg, *map(f.arg.__getitem__, argnums)), None


def linearize_grid(f, argnums):
  return f.grid + Grid.none(len(argnums))


linearize_p = core.Primitive("linearize")
def_operator(
  linearize_p,
  linearize_impl,
  linearize_spec,
  linearize_grid,
)


def _linearize_transpose_rule(t, f, *, argnums):
  # t: (primal, tangent) -> (primal_out, tangent_out)
  # f: (primal -> out)
  # seems we need to integrate over one of the inputs.
  # not very sure as this will go to infinity.
  raise NotImplementedError


def _linearize_jvp_rule(primals, tangents, *, argnums):
  # this is supposed to be an linear operator
  # because it consists of applying a nabla operator on the input function
  # and then product with an identity function.
  f_dot = tangents[0]
  return linearize_p.bind(f_dot, argnums=argnums)


jax.interpreters.ad.primitive_jvps[linearize_p] = _linearize_jvp_rule
jax.interpreters.ad.primitive_transposes[linearize_p
                                        ] = _linearize_transpose_rule


def constant(*, value, arg_shape):
  return constant_p.bind(value=value)


@operator
@lu.wrap_init
def constant_impl(*, value, arg_shape):

  @function
  @lu.wrap_init
  def const_value():
    return value

  return const_value


def constant_spec(*, value, arg_shape):
  return (ret_like(value), *arg_shape)


def constant_grid(*, value, arg_shape):
  return Grid.none(len(arg_shape))


constant_p = core.Primitive("constant")
def_operator(
  constant_p,
  constant_impl,
  constant_spec,
  constant_grid,
)


def partial(f, *, args, argnums=0):
  argnums = (argnums,) if isinstance(argnums, int) else argnums
  return partial_p.bind(f, args=args, argnums=argnums)


def partial_spec(f, *, args, argnums):
  assert len(argnums) == len(args)
  c_argnums = complement_argnums(argnums, num_args(f))
  return (f.ret, *(f.arg[i] for i in c_argnums)), None


def partial_grid(f, *, args, argnums):
  c_argnums = complement_argnums(argnums, num_args(f))
  return f.grid.take(c_argnums)


partial_p = core.Primitive("partial")
def_operator(
  partial_p,
  partial_impl,
  partial_spec,
  partial_grid,
)


def _partial_transpose_rule(t, f, *, args, argnums):
  argnums = (argnums,) if isinstance(argnums, int) else argnums
  cargnums = complement_argnums(argnums, num_args(f))
  assert ad.is_undefined_primal(f)
  wrt_f = broadcast_arg(t, f, cargnums)
  return (wrt_f,)


def _partial_jvp_rule(primals, tangents, *, args, argnums):
  f = primals[0]
  f_dot = tangents[0]
  jvps = []
  if not isinstance(f_dot, ad_util.Zero):
    jvp1 = partial(f_dot, args=args, argnums=argnums)
    jvps.append(jvp1)
  return partial(f, args=args, argnums=argnums), sum(jvps)


ad.primitive_jvps[partial_p] = _partial_jvp_rule
ad.primitive_transposes[partial_p] = _partial_transpose_rule


def invoke(f, *args):
  if len(args) != num_args(f):
    raise ValueError(
      f"Function {f} expects {num_args(f)} arguments, got {len(args)}."
    )
  return integrate(partial(f, args=args, argnums=tuple(range(len(args)))))


def binary_op(op, f1, f2, **params):
  op = functools.partial(op, **params)
  if not is_function(f1) and not is_function(f2):
    raise ValueError(
      "This will not happen because binary op is bind to at least one function."
    )
  elif is_function(f1) and not is_function(f2):
    if isinstance(f2, (tuple, list)):
      return compose(
        HashableFunction(lambda x: tree_map(op, x, f2), (op, f2)),
        f1,
      )
    else:
      return compose(
        HashableFunction(lambda x: tree_map(lambda y: op(y, f2), x), (op, f2)),
        f1,
      )
  elif is_function(f2) and not is_function(f1):
    if isinstance(f1, (tuple, list)):
      return compose(
        HashableFunction(lambda x: tree_map(op, f1, x), (op, f1)), f2
      )
    else:
      return compose(
        HashableFunction(lambda x: tree_map(lambda y: op(f1, y), x), (op, f1)),
        f2,
      )
  else:
    ret_f1, ret_f2 = SpecTree.from_ret(f1), SpecTree.from_ret(f2)
    is_pytree = lambda spec: isinstance(spec, (tuple, list))
    if (
      is_pytree(ret_f1) and is_pytree(ret_f2) and
      tree_structure(ret_f1) != tree_structure(ret_f2)
    ):
      raise ValueError(
        f"Cannot apply {op} on functions with different return structure. "
        f"Got {ret_f1} and {ret_f2}"
      )
    if SpecTree.from_args(f1) != SpecTree.from_args(f2):
      raise ValueError(
        f"Cannot apply {op} on functions with different arg specs. "
        f"Got {SpecTree.from_args(f1)} and {SpecTree.from_args(f2)}"
      )
    if is_pytree(ret_f1) and not is_pytree(ret_f2):
      return compose(
        HashableFunction(
          lambda tree, y: tree_map(lambda x: op(x, y), tree), (op,)
        ),
        f1,
        f2,
        share_inputs=True,
      )
    elif is_pytree(ret_f2) and not is_pytree(ret_f1):
      return compose(
        lambda x, tree: tree_map(lambda y: op(x, y), tree),
        f1,
        f2,
        share_inputs=True,
      )
    else:
      return compose(lambda x, y: tree_map(op, x, y), f1, f2, share_inputs=True)


def unary_op(op, f, **params):
  op = functools.partial(op, **params)
  if not is_function(f):
    raise ValueError(
      "This will not happen because unary op is bind to a function."
    )
  return compose(lambda x: tree_map(op, x), f, share_inputs=True)


funcs = {
  "numpy":
    {
      "negative": unary_op,
      "exp": unary_op,
      "log": unary_op,
      "sin": unary_op,
      "cos": unary_op,
      "tan": unary_op,
      "arcsin": unary_op,
      "arccos": unary_op,
      "arctan": unary_op,
      "sinh": unary_op,
      "cosh": unary_op,
      "tanh": unary_op,
      "arcsinh": unary_op,
      "arccosh": unary_op,
      "arctanh": unary_op,
      "square": unary_op,
      "sqrt": unary_op,
      "abs": unary_op,
      "sum": unary_op,
      "prod": unary_op,
      "mean": unary_op,
      "max": unary_op,
      "min": unary_op,
      "argmax": unary_op,
      "argmin": unary_op,
      "sign": unary_op,
      "add": binary_op,
      "subtract": binary_op,
      "multiply": binary_op,
      "divide": binary_op,
      "power": binary_op,
    },
  "nn": {
    "sigmoid": unary_op,
    "relu": unary_op,
  },  # noqa
  "lax": {
    "erf": unary_op,
  },  # noqa
}

numpy = SimpleNamespace()
lax = SimpleNamespace()
nn = SimpleNamespace()

for namespace, fns in funcs.items():
  ns = globals()[namespace]
  for name, nary_op in fns.items():
    jax_fn = getattr(getattr(jax, namespace), name)
    setattr(ns, name, functools.partial(nary_op, jax_fn))


def add(*fs):
  """Add functions together.

  Equivalent to:

  .. code-block:: python

    def add(*fs):
      def summed(*args):
        return sum([f(*args) for f in fs])
      return summed

  Args:
    x, y: functions to add together.
  Returns:
    A function that sums the outputs of the functions.
  """
  if len(fs) > 1:

    def add_values(*args):
      return tree_map(lambda *xs: sum(xs), *args)

    return compose(add_values, *fs, share_inputs=True)
  elif len(fs) == 1:
    return fs[0]


ad_util.jaxval_adders[types.FunctionType] = add
ad_util.jaxval_adders[function] = add

array_operators = {
  "neg": lambda x: numpy.negative(x),  # noqa
  "add": lambda x, y: numpy.add(x, y),
  "radd": lambda y, x: numpy.add(x, y),
  "sub": lambda x, y: numpy.subtract(x, y),
  "rsub": lambda y, x: numpy.subtract(x, y),
  "truediv": lambda x, y: numpy.divide(x, y),
  "rtruediv": lambda y, x: numpy.divide(x, y),
  "mul": lambda x, y: numpy.multiply(x, y),
  "rmul": lambda y, x: numpy.multiply(x, y),
  "pow": lambda x, y: numpy.power(x, y),
  "rpow": lambda y, x: numpy.power(x, y),
}


def call(obj, *args, **params):
  args = tuple(
    jnp.asarray(a) if isinstance(a, (float, int)) else a for a in args
  )
  # check all inputs are correct
  arg_spec = tuple(
    SpecTree.from_ret(a) if is_function(a) else SpecTree.from_value(a)
    for a in args
  )
  if arg_spec != obj.arg_spec:
    raise ValueError(
      f"Spec mismatch: function {obj.f} is marked as "
      f"requiring {obj.arg_spec} as input, got {arg_spec}."
    )
  # separate function and non-function arguments
  non_function_args = []
  non_function_index = []
  function_args = []
  for i, arg in enumerate(args):
    if is_function(arg):
      function_args.append(arg)
    else:
      non_function_args.append(arg)
      non_function_index.append(i)

  # if all are non functions, call invoke
  if len(function_args) == 0:
    if isinstance(obj, function):
      # if this is a function instance, directly invoke it.
      ret = obj._call(*args)  # noqa
    else:
      # if this is abstract, use the invoke operator instead.
      ret = invoke(obj, *args)
    if SpecTree.from_value(ret) != obj.ret_spec:
      raise ValueError(
        f"Spec mismatch: function {obj.f} is marked as "
        f"returning {obj.ret_spec}, got {SpecTree.from_value(ret)}."
      )
    return ret
  # if there are both function and non-function arguments
  # first bind the non-function arguments
  # then compose with the function arguments
  elif len(non_function_args) != 0:
    f = partial(obj, args=non_function_args, argnums=tuple(non_function_index))
  # if all are function arguments, just compose
  else:
    f = obj
  return compose(f, *function_args, **params)


def _set_function_operators(cls, operators):
  cls.__call__ = call
  for op_name, op in operators.items():
    setattr(cls, f"__{op_name}__", op)


def _set_general_array_operators(cls, operators):
  cls.__call__ = call
  for op_name, op in operators.items():
    setattr(cls, f"_{op_name}", staticmethod(op))


def _set_tracer_call():

  def call(self, *args):
    if isinstance(self.aval, GeneralArray):
      return invoke(self, *args)
    else:
      raise ValueError("Calling an array")

  jax.core.Tracer.__call__ = call


_set_function_operators(function, array_operators)
_set_general_array_operators(GeneralArray, array_operators)
_set_tracer_call()
