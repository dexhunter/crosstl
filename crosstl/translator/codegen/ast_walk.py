"""Iterative pre-order walk over every object reachable from an AST value.

The code generators used to walk with nested ``yield from`` generators. Each
yielded node then passed through every enclosing generator frame, so a walk cost
the node count times the nesting depth, and a deep expression could exceed the
recursion limit. This walk keeps an explicit stack of iterators instead and
yields exactly the same objects in exactly the same order:

- ``None``, ``str``, ``int``, ``float`` and ``bool`` values are skipped;
- ``dict`` values, and the items of a ``list``, ``tuple`` or ``set``, are
  visited in iteration order, and the containers themselves are not yielded;
- any other object is yielded once (by identity), then its ``vars()`` values are
  visited in attribute order.

Child iterators are created only when the walk resumes after yielding their
owner, as before, so a caller may still change the attributes of the node it has
just received.
"""

from typing import Any, Iterator

_SCALARS = (str, int, float, bool)
_SEQUENCES = (list, tuple, set)


def walk_ast_values(root: Any) -> Iterator[Any]:
    """Yield every non-container object reachable from ``root`` once, in pre-order."""
    # Each value type is classified once per walk; empty containers are skipped.
    skip, mapping, sequence, node = range(4)
    kinds = {type(None): skip}
    visited = set()
    stack = [iter((root,))]
    push = stack.append
    while stack:
        for value in stack[-1]:
            kind = kinds.get(type(value))
            if kind is None:
                if isinstance(value, _SCALARS):
                    kind = skip
                elif isinstance(value, dict):
                    kind = mapping
                elif isinstance(value, _SEQUENCES):
                    kind = sequence
                else:
                    kind = node
                kinds[type(value)] = kind
            if kind is skip:
                continue
            if kind is node:
                value_id = id(value)
                if value_id in visited:
                    continue
                visited.add(value_id)
                yield value
                # Walk the attribute dict as a mapping; test it before
                # building a values view so empty nodes allocate nothing.
                value = getattr(value, "__dict__", None)
                kind = mapping
            if value:
                push(iter(value.values() if kind is mapping else value))
                break
        else:
            stack.pop()
