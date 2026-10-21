"""Generate float64 validation kernels from the same Warp function definitions.

Only Warp-decorated functions are transformed. Their scalar, vector, matrix and
quaternion types and floating literals are promoted together; branches, integer
indices and arithmetic expressions stay the same. Host-side classes are shared.
"""

import ast
import inspect
import linecache
import sys
import types

import warp as wp


_TYPES = {float: wp.float64, wp.float32: wp.float64, wp.vec2: wp.vec2d,
          wp.vec3: wp.vec3d, wp.mat33: wp.mat33d, wp.quat: wp.quatd,
          wp.spatial_vector: wp.spatial_vectord}
_NAMES = {'float32': 'float64', 'vec2': 'vec2d', 'vec3': 'vec3d', 'mat33': 'mat33d', 'quat': 'quatd',
          'spatial_vector': 'spatial_vectord'}
_MODULES = {}


def double_type(dtype):
    return _TYPES.get(dtype, dtype)


class _Promote(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id == 'float':
            return ast.copy_location(ast.Attribute(value=ast.Name(id='wp', ctx=ast.Load()),
                                                   attr='float64', ctx=node.ctx), node)
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == 'wp' and node.attr in _NAMES:
            node.attr = _NAMES[node.attr]
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, float):
            return ast.copy_location(ast.Call(func=ast.Attribute(value=ast.Name(id='wp', ctx=ast.Load()),
                                                                 attr='float64', ctx=ast.Load()),
                                              args=[node], keywords=[]), node)
        return node


def _warp_definition(node):
    if not isinstance(node, ast.FunctionDef):
        return False
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                and target.value.id == 'wp' and target.attr in ('func', 'kernel', 'func_grad', 'func_replay')):
            return True
    return False


def _module(original):
    if original.__name__ in _MODULES:
        return _MODULES[original.__name__]
    generated = types.ModuleType(original.__name__ + '_float64')
    _MODULES[original.__name__] = generated
    namespace = generated.__dict__
    namespace.update(original.__dict__)
    namespace['__name__'] = original.__name__ + '_float64'
    sys.modules[namespace['__name__']] = generated
    for name, value in list(namespace.items()):
        if isinstance(value, types.ModuleType) and value.__name__.startswith('blademaster_warp'):
            namespace[name] = _module(value)
        elif hasattr(value, 'adj') and hasattr(value, 'func') and value.func is not None:
            parent = sys.modules.get(value.func.__module__)
            if parent is not None and parent is not original:
                namespace[name] = getattr(_module(parent), value.func.__name__)
    parsed = ast.parse(inspect.getsource(original))
    definitions = [_Promote().visit(node) for node in parsed.body if _warp_definition(node)]
    source = ast.unparse(ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))) + '\n'
    filename = '<' + namespace['__name__'] + '>'
    namespace['__file__'] = filename
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, 'exec'), namespace)
    return generated


def double_kernel(kernel):
    original = sys.modules[kernel.func.__module__]
    return getattr(_module(original), kernel.func.__name__)


def launch_double(kernel, dim, inputs=None, outputs=None, **kwargs):
    promoted = double_kernel(kernel)
    values = list(inputs or ()) + list(outputs or ())
    for index, (value, argument) in enumerate(zip(values, promoted.adj.args)):
        expected = argument.type
        if hasattr(expected, '_wp_scalar_type_') and expected._wp_scalar_type_ == wp.float64:
            if not isinstance(value, expected):
                values[index] = expected(*value)
    return wp.launch(promoted, dim, inputs=values, **kwargs)
