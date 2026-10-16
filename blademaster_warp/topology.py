"""Mechanical connected components from the reference grid and cut codes."""
import numpy as np


def compatible(a, b):
    a, b = int(a), int(b)
    while a or b:
        x, y = a % 3, b % 3
        if x and y and x != y:
            return False
        a //= 3; b //= 3
    return True


def components(nodes, weights, codes):
    nodes, weights, codes = np.asarray(nodes), np.asarray(weights), np.asarray(codes)
    n = len(codes)
    parent = np.arange(n)
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def join(a,b):
        a,b=find(a),find(b)
        if a != b: parent[max(a,b)] = min(a,b)
    groups={}
    for p in range(n):
        for node in nodes[p][weights[p] > 0]:
            by_code=groups.setdefault(int(node),{})
            code=int(codes[p])
            if code in by_code: join(p,by_code[code])
            else: by_code[code]=p
    for by_code in groups.values():
        items=list(by_code.items())
        for i,(a,pa) in enumerate(items):
            for b,pb in items[i+1:]:
                if compatible(a,b): join(pa,pb)
    roots=np.array([find(p) for p in range(n)])
    _,labels,counts=np.unique(roots,return_inverse=True,return_counts=True)
    return labels,counts
