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
from jax import core
from graphviz import Digraph
import itertools
import weakref
from functools import lru_cache, wraps


def weak_lru(maxsize=128, typed=False):
  'LRU Cache decorator that keeps a weak reference to "self"'

  def wrapper(func):

    @lru_cache(maxsize, typed)
    def _func(_self, *args, **kwargs):
      return func(_self(), *args, **kwargs)

    @wraps(func)
    def inner(self, *args, **kwargs):
      return _func(weakref.ref(self), *args, **kwargs)

    return inner

  return wrapper


def complement_argnums(argnums, num_args):
  argnums = (argnums,) if isinstance(argnums, int) else argnums
  complement = []
  a = 0
  for i in range(num_args):
    if a < len(argnums) and i == argnums[a]:
      a += 1
    else:
      complement.append(i)
  return tuple(complement)


# visualization
# adopted from
# https://gist.github.com/niklasschmitz/559a1f717f3535db0e26d0edccad0b46

styles = {
  'const': dict(style='filled', color='goldenrod1'),
  'invar': dict(color='mediumspringgreen', style='filled'),
  'outvar': dict(style='filled,dashed', fillcolor='indianred1', color='black'),
  'op_node': dict(shape='box', color='lightskyblue', style='filled'),
  'intermediate': dict(style='filled', color='cornflowerblue')
}


def _jaxpr_graph(jaxpr):
  id_names = (f'id{id}' for id in itertools.count())
  graph = Digraph(engine='dot')
  graph.attr(size='6,10!')
  for v in jaxpr.constvars:
    graph.node(
      str(v),
      core.raise_to_shaped(v.aval).str_short(), styles['const']
    )
  for v in jaxpr.invars:
    graph.node(str(v), v.aval.str_short(), styles['invar'])
  for eqn in jaxpr.eqns:
    for v in eqn.invars:
      if isinstance(v, core.Literal):
        graph.node(
          str(id(v.val)),
          core.raise_to_shaped(core.get_aval(v.val)).str_short(),
          styles['const']
        )
    if eqn.primitive.multiple_results:
      id_name = next(id_names)
      graph.node(id_name, str(eqn.primitive), styles['op_node'])
      for v in eqn.invars:
        graph.edge(
          str(id(v.val) if isinstance(v, core.Literal) else v), id_name
        )
      for v in eqn.outvars:
        graph.node(str(v), v.aval.str_short(), styles['intermediate'])
        graph.edge(id_name, str(v))
    else:
      outv, = eqn.outvars
      graph.node(str(outv), str(eqn.primitive), styles['op_node'])
      for v in eqn.invars:
        graph.edge(
          str(id(v.val) if isinstance(v, core.Literal) else v), str(outv)
        )
  for i, v in enumerate(jaxpr.outvars):
    outv = 'out_' + str(i)
    graph.node(outv, outv, styles['outvar'])
    graph.edge(str(v), outv)
  return graph


def visualize_jaxpr(fn):

  def wrapper(*args, **kwargs):
    jaxpr = jax.make_jaxpr(fn)(*args, **kwargs).jaxpr
    g = _jaxpr_graph(jaxpr)
    filename = str(hash(jaxpr))
    g.view(filename=filename, directory="/tmp", cleanup=True)
    print(filename)
    return fn(*args, **kwargs)

  return wrapper
