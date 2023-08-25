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

# jaxpr to graphviz
# https://gist.github.com/niklasschmitz/559a1f717f3535db0e26d0edccad0b46

styles = {
  "const": {
    "style": "filled",
    "color": "goldenrod1",
    "fontname": "courier",
  },
  "invar":
    {
      "style": "filled,dashed",
      "fillcolor": "mediumspringgreen",
      "color": "black",
      "fontname": "courier",
    },
  "outvar":
    {
      "style": "filled,dashed",
      "fillcolor": "indianred1",
      "color": "black",
      "fontname": "courier",
    },
  "op_node": {
    "style": "filled",
    "shape": "box",
    "color": "lightskyblue",
  },
  "intermediate":
    {
      "style": "filled",
      "color": "cornflowerblue",
      "fontname": "courier",
    },
}


def _jaxpr_graph(closed_jaxpr):
  id_names = (f"id{id}" for id in itertools.count())
  graph = Digraph(engine="dot")
  graph.attr(size="6,10!")
  for v, val in zip(closed_jaxpr.jaxpr.constvars, closed_jaxpr.consts):
    graph.node(
      str(v),
      # v.aval.str_short(),
      core.raise_to_shaped(core.get_aval(val)).str_short(),
      _attributes=styles["const"],
    )
  for v, val in zip(closed_jaxpr.jaxpr.invars, closed_jaxpr.in_avals):
    graph.node(
      str(v),
      val.str_short(),
      # v.aval.str_short(),
      _attributes=styles["invar"],
    )
  for eqn in closed_jaxpr.eqns:
    for v in eqn.invars:
      if isinstance(v, core.Literal):
        graph.node(
          str(id(v.val)),
          core.raise_to_shaped(core.get_aval(v.val)).str_short(),
          _attributes=styles["const"],
        )
    if eqn.primitive.multiple_results:
      id_name = next(id_names)
      graph.node(id_name, str(eqn.primitive), _attributes=styles["op_node"])
      for v in eqn.invars:
        graph.edge(
          str(id(v.val) if isinstance(v, core.Literal) else v), id_name
        )
      for v in eqn.outvars:
        graph.node(
          str(v),
          v.aval.str_short(),
          _attributes=styles["intermediate"],
        )
        graph.edge(id_name, str(v))
    else:
      outv, = eqn.outvars
      graph.node(
        str(outv),
        str(eqn.primitive),
        _attributes=styles["op_node"],
      )
      for i, v in enumerate(eqn.invars):
        headlabel = str(i) if len(eqn.invars) > 1 else None
        graph.edge(
          str(id(v.val) if isinstance(v, core.Literal) else v),
          str(outv),
          headlabel=headlabel,
          labelfontcolor="blue",
        )
  for i, v in enumerate(closed_jaxpr.jaxpr.outvars):
    outv = "out_" + str(i)
    graph.node(outv, outv, _attributes=styles["outvar"])
    graph.edge(str(v), outv)
  return graph


def jaxpr_graph(fun, *args):
  jaxpr = jax.make_jaxpr(fun)(*args).jaxpr
  return _jaxpr_graph(jaxpr)


def grad_graph(fun, *args):
  _, fun_vjp = jax.vjp(fun, *args)
  jaxpr = fun_vjp.args[0].func.args[1]
  return _jaxpr_graph(jaxpr)
