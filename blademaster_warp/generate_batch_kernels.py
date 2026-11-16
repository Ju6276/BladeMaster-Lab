"""Generate world-indexed equivalents of unchanged scalar physics kernels.

Array indexing, launch indices and per-world material parameter loads change.
Generated files are reviewable source, not an alternative constitutive model.
"""

import ast
from pathlib import Path

NAMES = ("material", "contact", "tool", "blade_constraint")
ROOT = Path(__file__).parent


def is_array(annotation):
    return (
        isinstance(annotation, ast.Call)
        and isinstance(annotation.func, ast.Attribute)
        and annotation.func.value.id == "wp"
        and annotation.func.attr.startswith("array")
    )


def generate():
    trees = {n: ast.parse((ROOT / (n + ".py")).read_text()) for n in NAMES}
    functions = {
        f.name
        for t in trees.values()
        for f in t.body
        if isinstance(f, ast.FunctionDef)
        and any(is_array(a.annotation) for a in f.args.args)
        and any(isinstance(d, ast.Attribute) and d.attr == "func" for d in f.decorator_list)
    }
    for name, tree in trees.items():
        out = []
        for item in tree.body:
            if isinstance(item, ast.ClassDef):
                continue
            if isinstance(item, ast.ImportFrom) and item.level and item.module in NAMES:
                item.module = "batch_" + item.module
            if not isinstance(item, ast.FunctionDef):
                out.append(item)
                continue
            kernel = any(isinstance(d, ast.Attribute) and d.attr == "kernel" for d in item.decorator_list)
            func = any(isinstance(d, ast.Attribute) and d.attr == "func" for d in item.decorator_list)
            if not (kernel or func):
                continue
            arrays = {a.arg for a in item.args.args if is_array(a.annotation)}
            if not arrays:
                out.append(item)
                continue
            for a in item.args.args:
                if a.arg in arrays:
                    an = a.annotation
                    dims = int(an.func.attr[5]) if an.func.attr != "array" else 1
                    for kw in an.keywords:
                        if kw.arg == "ndim":
                            dims = kw.value.value
                    an.func.attr = "array"
                    an.keywords = [kw for kw in an.keywords if kw.arg != "ndim"] + [
                        ast.keyword(arg="ndim", value=ast.Constant(dims + 1))
                    ]
            tid_dims = 1
            for n in ast.walk(item):
                if (
                    isinstance(n, ast.Assign)
                    and isinstance(n.value, ast.Call)
                    and ast.unparse(n.value.func) == "wp.tid"
                    and isinstance(n.targets[0], ast.Tuple)
                ):
                    tid_dims = len(n.targets[0].elts)

            class Transform(ast.NodeTransformer):
                def visit_Subscript(self, n):
                    n = self.generic_visit(n)
                    if isinstance(n.value, ast.Name) and n.value.id in arrays:
                        indices = list(n.slice.elts) if isinstance(n.slice, ast.Tuple) else [n.slice]
                        n.slice = ast.Tuple(
                            elts=[ast.Name(id="world", ctx=ast.Load()), *indices], ctx=ast.Load()
                        )
                    elif (
                        isinstance(n.value, ast.Attribute)
                        and n.value.attr == "shape"
                        and isinstance(n.value.value, ast.Name)
                        and n.value.value.id in arrays
                    ):
                        n.slice = ast.BinOp(n.slice, ast.Add(), ast.Constant(1))
                    return n

                def visit_Call(self, n):
                    if kernel and ast.unparse(n.func) == "wp.tid":
                        names = [ast.Name(id="_batch_tid" + str(i), ctx=ast.Load()) for i in range(tid_dims)]
                        return names[0] if tid_dims == 1 else ast.Tuple(elts=names, ctx=ast.Load())
                    n = self.generic_visit(n)
                    if isinstance(n.func, ast.Name) and n.func.id in functions:
                        n.args.insert(0, ast.Name(id="world", ctx=ast.Load()))
                    elif (
                        isinstance(n.func, ast.Attribute)
                        and isinstance(n.func.value, ast.Name)
                        and n.func.value.id == "wp"
                        and n.func.attr.startswith("atomic_")
                        and n.args
                        and isinstance(n.args[0], ast.Name)
                        and n.args[0].id in arrays
                    ):
                        n.args.insert(1, ast.Name(id="world", ctx=ast.Load()))
                    return n

            item = Transform().visit(item)
            # Functions still receive scalar coefficients. Only kernel boundaries
            # load the world's coefficients before executing the original body.
            parameters = []
            if kernel and name in ("material", "contact", "profile_contact"):
                for argument in item.args.args:
                    if (argument.arg in ("mass", "mu", "lam", "mu_maxwell", "threshold")
                            and isinstance(argument.annotation, ast.Name)
                            and argument.annotation.id == "float"):
                        label = argument.arg
                        argument.arg = "_world_" + label
                        argument.annotation = ast.parse("wp.array(dtype=float)", mode="eval").body
                        parameters.extend(ast.parse(label + " = _world_" + label + "[world]").body)
            item.body = parameters + item.body
            if kernel:
                item.args.args.insert(
                    0, ast.arg(arg="enabled", annotation=ast.parse("wp.array(dtype=int)", mode="eval").body)
                )
                names = ", ".join(["world"] + ["_batch_tid" + str(i) for i in range(tid_dims)])
                item.body = (
                    ast.parse(names + " = wp.tid()\nif enabled[world] == 0:\n    return").body + item.body
                )
            else:
                item.args.args.insert(0, ast.arg(arg="world", annotation=ast.Name(id="int", ctx=ast.Load())))
            out.append(item)
        result = ast.Module(body=out, type_ignores=[])
        ast.fix_missing_locations(result)
        (ROOT / ("batch_" + name + ".py")).write_text(
            "# Generated by generate_batch_kernels.py; physics expressions retained.\n"
            + ast.unparse(result)
            + "\n"
        )


if __name__ == "__main__":
    generate()
