"""CPU reference linked-volume / Hermite / local dual-contouring surface.

Implements the construction described in BladeMaster III-F independently.
Only actual finite edge sweeps sever links. No future path or mechanical
component mesh is used. This is one-way visualization, never collision input.

Two explicit implementation choices, unspecified in the paper:
* bounded least-squares QEF per local region;
* affine-exact P2G position extrapolation followed by fixed trilinear weights
  in compatible material fields. This is not a copied author embedding kernel.
"""
from itertools import product
from dataclasses import dataclass
import numpy as np
from scipy.optimize import lsq_linear
from .geometry import levelset
from .topology import compatible

OFFSETS=np.array(list(product((0,1),repeat=3)),int)
LOCAL_EDGES=[(a,b) for a in range(8) for b in range(a+1,8) if np.abs(OFFSETS[a]-OFFSETS[b]).sum()==1]


def segment_triangle(a,b,tri):
    """Vectorized finite segment/triangle intersections, including endpoints."""
    direction=b-a;e1=tri[1]-tri[0];e2=tri[2]-tri[0]
    h=np.cross(direction,e2);det=h@e1
    valid=np.abs(det)>1.e-15
    inv=np.divide(1.,det,out=np.zeros_like(det),where=valid)
    s=a-tri[0];u=np.einsum('ij,ij->i',s,h)*inv
    q=np.cross(s,e1);v=np.einsum('ij,ij->i',direction,q)*inv;t=(q@e2)*inv
    return valid&(u>=-1.e-9)&(v>=-1.e-9)&(u+v<=1+1.e-9)&(t>=-1.e-9)&(t<=1+1.e-9),t


def moving_segment_hits(a0,b0,a1,b1,edge0,edge1):
    """CCD between linearly moving material links and a finite blade edge.

    Testing only current links against the blade's swept triangle can miss a
    cut when the material itself moves across the edge between observations.
    Coplanarity is a cubic in time; solve it, then test both segment bounds.
    This affects surface topology only, never the mechanical cutting labels.
    """
    hit=np.zeros(len(a0),bool);alpha=np.zeros(len(a0));times=np.zeros(len(a0))
    c0,d0=edge0;c1,d1=edge1;v0=d0-c0;dv=(d1-c1)-v0
    blade_lo=np.minimum(edge0.min(0),edge1.min(0));blade_hi=np.maximum(edge0.max(0),edge1.max(0))
    for i in range(len(a0)):
        endpoints=np.array([a0[i],b0[i],a1[i],b1[i]])
        if np.any(endpoints.max(0)<blade_lo-1.e-10) or np.any(endpoints.min(0)>blade_hi+1.e-10):continue
        u0=b0[i]-a0[i];du=b1[i]-a1[i]-u0
        w0=c0-a0[i];dw=c1-a1[i]-w0
        cros0=np.cross(u0,v0);cros1=np.cross(du,v0)+np.cross(u0,dv);cros2=np.cross(du,dv)
        coeff=np.array([cros0@w0,cros0@dw+cros1@w0,cros1@dw+cros2@w0,cros2@dw])
        threshold=max(abs(coeff).max()*1.e-12,1.e-25)
        nonzero=np.flatnonzero(abs(coeff)>threshold)
        candidates=[]
        if len(nonzero):
            roots=np.roots(coeff[:nonzero[-1]+1][::-1])
            candidates=[float(r.real) for r in roots if abs(r.imag)<1.e-8 and -1.e-9<=r.real<=1+1.e-9]
        else:
            # Coplanar for the whole interval: intersection starts/ends when
            # an endpoint crosses the other supporting line. Each cross-product
            # component is at most quadratic. Include endpoints and intervals.
            candidates=[0.,1.]
            for p0,dp,q0,dq in [(w0,dw,u0,du),(w0+v0,dw+dv,u0,du),
                                 (-w0,-dw,v0,dv),(u0-w0,du-dw,v0,dv)]:
                cc=np.array([np.cross(p0,q0),np.cross(dp,q0)+np.cross(p0,dq),np.cross(dp,dq)])
                for axis in range(3):
                    nz=np.flatnonzero(abs(cc[:,axis])>1.e-25)
                    if len(nz):
                        for root in np.roots(cc[:nz[-1]+1,axis][::-1]):
                            if abs(root.imag)<1.e-8 and 0<=root.real<=1:candidates.append(float(root.real))
            candidates=sorted(set(candidates));candidates+=list((np.array(candidates[:-1])+candidates[1:])/2)
        for t in sorted(candidates):
            t=float(np.clip(t,0,1));u=u0+t*du;v=v0+t*dv;w=w0+t*dw
            matrix=np.column_stack([u,-v]);uv=np.linalg.lstsq(matrix,w,rcond=None)[0]
            if (np.linalg.norm(matrix@uv-w)<=1.e-9 and np.all(uv>=-1.e-8) and np.all(uv<=1+1.e-8)):
                hit[i]=True;alpha[i]=uv[0];times[i]=t;break
    return hit,alpha,times


@dataclass
class Mesh:
    reference:np.ndarray
    triangles:np.ndarray
    codes:np.ndarray
    cells:np.ndarray
    weights:np.ndarray


class LinkedSurface:
    def __init__(self,size,spacing,shape='block'):
        self.size=np.array(size,float);self.h=float(spacing);self.shape=shape
        # Half-cell padding keeps box boundaries off primal nodes and encloses
        # the original surface. It is not a motion/workspace constraint.
        self.origin=np.array([-size[0]/2,-size[1]/2,0])-spacing/2
        self.dims=np.ceil(self.size/spacing).astype(int)+2
        ijk=np.array(list(product(*(range(n) for n in self.dims))))
        self.reference=self.origin+ijk*spacing
        sdf=levelset(self.reference,size,shape)
        self.inside=sdf<0
        self.codes=np.zeros(len(ijk),np.int64)
        pairs=[];self.axes=[]
        for a,idx in enumerate(ijk):
            for axis in range(3):
                nxt=idx.copy();nxt[axis]+=1
                if nxt[axis]<self.dims[axis]:
                    pairs.append((a,np.ravel_multi_index(nxt,self.dims)));self.axes.append(axis)
        self.edges=np.array(pairs,int);self.axes=np.array(self.axes)
        self.broken=np.zeros(len(pairs),bool)
        self.boundary=self.inside[self.edges[:,0]]!=self.inside[self.edges[:,1]]
        self.hermite={}
        for e in np.flatnonzero(self.boundary):
            a,b=self.reference[self.edges[e]];lo,hi=0.,1.
            for _ in range(35):
                t=(lo+hi)/2
                if (levelset(a+t*(b-a),size,shape)<0)==(sdf[self.edges[e,0]]<0):lo=t
                else:hi=t
            p=a+(lo+hi)/2*(b-a);eps=spacing*1.e-4
            n=np.array([(levelset(p+eps*np.eye(3)[j],size,shape)-levelset(p-eps*np.eye(3)[j],size,shape))/(2*eps) for j in range(3)])
            n/=max(np.linalg.norm(n),1.e-15)
            self.hermite[int(e)]=(p,n)
        lookup={tuple(pair):e for e,pair in enumerate(self.edges)}
        self.cells=[];self.cell_edges=[];self.incident=[[] for _ in pairs]
        for idx in product(*(range(n-1) for n in self.dims)):
            corners=np.ravel_multi_index((OFFSETS+idx).T,self.dims)
            if not self.inside[corners].any():continue
            es=[lookup[tuple(sorted((corners[a],corners[b])))] for a,b in LOCAL_EDGES]
            ci=len(self.cells);self.cells.append(corners);self.cell_edges.append(es)
            for e in es:self.incident[e].append(ci)
        self.cells=np.array(self.cells);self.cell_edges=np.array(self.cell_edges)
        self.regions={};self.dirty=set(range(len(self.cells)));self.updated_cells=0
        self._mesh_cache=None

    def sweep(self,edge_before,edge_after,current_nodes=None,cut_index=0,normal_world=None,node_deformation=None,
              previous_nodes=None,previous_deformation=None,ccd=moving_segment_hits):
        """Sever links hit by the actual swept edge; returning to a cut cannot heal it.

        current_nodes must be the current compatible field embedding of the
        reference nodes. Identity is useful for geometry-only unit tests.
        """
        if not 0<=cut_index<19:raise ValueError('Cut code storage supports 19 events')
        x=self.reference if current_nodes is None else np.asarray(current_nodes)
        quad=np.asarray([edge_before[0],edge_before[1],edge_after[1],edge_after[0]])
        candidates=np.flatnonzero(self.inside[self.edges].all(axis=1)&~self.broken)
        if not len(candidates):return 0
        aa,bb=x[self.edges[candidates,0]],x[self.edges[candidates,1]]
        hits=[]
        batches=[]
        if previous_nodes is not None:
            old=np.asarray(previous_nodes)
            hit,alpha,time=ccd(old[self.edges[candidates,0]],old[self.edges[candidates,1]],
                aa,bb,np.asarray(edge_before),np.asarray(edge_after))
            n=np.asarray(normal_world,float) if normal_world is not None else np.cross(edge_before[1]-edge_before[0],edge_after[0]-edge_before[0])
            if np.linalg.norm(n)<1.e-15:raise ValueError('A normal is required for a stationary blade with moving material')
            n=n/np.linalg.norm(n)
            batches.append((candidates[hit],alpha[hit],time[hit],n))
        for tri in (() if previous_nodes is not None else (quad[[0,1,2]],quad[[0,2,3]])):
            n=np.cross(tri[1]-tri[0],tri[2]-tri[0]);norm=np.linalg.norm(n)
            if norm<1.e-15:continue
            n/=norm
            if normal_world is not None:
                n=np.asarray(normal_world,float)/np.linalg.norm(normal_world)
            hit,t=segment_triangle(aa,bb,tri)
            batches.append((candidates[hit],t[hit],np.ones(hit.sum()),n))
        for batch,alphas,times,n in batches:
            for e,alpha,t in zip(batch,alphas,times):
                if self.broken[e]:continue
                a,b=self.edges[e];p=(1-alpha)*self.reference[a]+alpha*self.reference[b]
                # Pull normal back using this edge's local motion approximation.
                # Full material pullback is provided by embedding_fn below in
                # dynamic use; reference-only tests use identity deformation.
                nref=n.copy()
                if node_deformation is not None:
                    f=(1-alpha)*node_deformation[a]+alpha*node_deformation[b]
                    if previous_deformation is not None:
                        oldf=(1-alpha)*previous_deformation[a]+alpha*previous_deformation[b]
                        f=(1-t)*oldf+t*f
                    nref=f.T@n
                    nref/=np.linalg.norm(nref)
                self.hermite[int(e)]=(p,nref)
                self.broken[e]=True;hits.append(int(e));self.dirty.update(self.incident[e])
                for node in (a,b):
                    if self.codes[node]//(3**cut_index)%3==0:
                        point=x[node] if previous_nodes is None else (1-t)*previous_nodes[node]+t*x[node]
                        blade_point=(1-t)*np.asarray(edge_before)[0]+t*np.asarray(edge_after)[0]
                        side=1 if np.dot(point-blade_point,n)<0 else 2
                        self.codes[node]+=side*3**cut_index
        # A node's new code changes every incident region using that node, not
        # just cells adjacent to the severed edge.
        if hits:
            changed=set(self.edges[hits].ravel())
            self.dirty.update(i for i,c in enumerate(self.cells) if any(int(v) in changed for v in c))
        return len(hits)

    def _update_cell(self,ci):
        corners=self.cells[ci];es=self.cell_edges[ci]
        active=set(np.flatnonzero(self.inside[corners]).tolist());regions=[]
        while active:
            region={active.pop()};todo=list(region)
            while todo:
                a=todo.pop()
                for e,(i,j) in zip(es,LOCAL_EDGES):
                    if self.broken[e]:continue
                    b=j if i==a else i if j==a else -1
                    if b in active:active.remove(b);region.add(b);todo.append(b)
            points=[];normals=[]
            # At a crack front both ends of a severed edge may still connect
            # around intact edges within the cell. That connected region still
            # needs a Hermite representative; omitting it leaves an open front.
            for e,(a,b) in zip(es,LOCAL_EDGES):
                if e in self.hermite and (((a in region)!=(b in region)) or
                        (self.broken[e] and (a in region or b in region))):
                    p,n=self.hermite[e];points.append(p);normals.append(n)
            if not points:continue
            points=np.array(points);normals=np.array(normals)
            centre=points.mean(0);rhs=np.einsum('ij,ij->i',normals,points-centre)
            # Solve offset relative to mass point to avoid drifting in the QEF
            # nullspace; bounded solve is only needed if unconstrained is outside.
            offset=np.linalg.lstsq(normals,rhs,rcond=None)[0]
            lower=self.reference[corners].min(0);upper=lower+self.h
            p=centre+offset
            if ((p<lower-1.e-10)|(p>upper+1.e-10)).any():
                p=centre+lsq_linear(normals,rhs,bounds=(lower-centre-1.e-12,upper-centre+1.e-12),tol=1.e-10).x
            local=(p-lower)/self.h
            weights=np.where(OFFSETS==1,local,1-local).prod(1)
            # Combine compatible assigned digits in this local graph region.
            code=0
            for k in range(19):
                digits={int(self.codes[corners[a]])//(3**k)%3 for a in region}-{0}
                if len(digits)==1:code+=digits.pop()*3**k
            regions.append((set(int(corners[a]) for a in region),p,code,weights))
        self.regions[ci]=regions

    def mesh(self,use_cache=True):
        self.updated_cells=len(self.dirty)
        # This caches reference topology/QEF coordinates only, never deformed
        # world vertices. A new cut invalidates it through the dirty-cell set;
        # surface_runtime still embeds the returned reference mesh every frame.
        if use_cache and not self.dirty and self._mesh_cache is not None:
            return self._mesh_cache
        self._mesh_cache=None
        for ci in self.dirty:self._update_cell(ci)
        self.dirty.clear()
        vertices=[];codes=[];cells=[];weights=[];lookup={}
        for ci,regions in sorted(self.regions.items()):
            for nodes,p,code,w in regions:
                vi=len(vertices);vertices.append(p);codes.append(code);cells.append(self.cells[ci]);weights.append(w)
                for node in nodes:lookup[ci,node]=vi
        faces=[]
        for e in np.flatnonzero(self.boundary|self.broken):
            a,b=self.edges[e];axis=self.axes[e];other=[j for j in range(3) if j!=axis]
            centre=self.reference[a]
            cs=sorted(self.incident[e],key=lambda ci:np.arctan2(*(self.reference[self.cells[ci]].mean(0)-centre)[other][::-1]))
            for node in (a,b):
                if not self.inside[node]:continue
                indices=[lookup[ci,node] for ci in cs if (ci,node) in lookup]
                indices=list(dict.fromkeys(indices))
                if len(indices)<3:continue
                # Cell centres are ordered counterclockwise in the two axes
                # perpendicular to the primal edge. Use that topological
                # orientation: QEF vertices can coincide at a flat cut, making
                # their cross-product zero (or roundoff with arbitrary sign).
                axis_orientation=-1 if axis==1 else 1
                sign=1 if node==a else -1
                if axis_orientation*sign<0:indices=indices[::-1]
                for j in range(1,len(indices)-1):
                    tri=[indices[0],indices[j],indices[j+1]]
                    if len(set(tri))==3:faces.append(tri)
        self._mesh_cache=Mesh(np.asarray(vertices).reshape(-1,3),np.asarray(faces,int).reshape(-1,3),np.asarray(codes),
                              np.asarray(cells,int).reshape(-1,8),np.asarray(weights).reshape(-1,8))
        return self._mesh_cache


class MaterialEmbedding:
    """Compatible nodal position fields; affine-exact and one-way.

    Reference support is fixed, not a nearest-current-particle mesh warp.
    Empty nodes use an explicitly declared compatible affine continuation.
    Missing support even in the next reference ring is an error, never a frozen face.
    """
    def __init__(self,reference_nodes,particle_reference,spacing,grid_origin=None,grid_cells=None):
        from scipy.spatial import cKDTree
        self.nodes=np.asarray(reference_nodes);self.ref=np.asarray(particle_reference);self.h=spacing
        self.material_grid=None
        if grid_origin is not None:
            self.grid_origin=np.asarray(grid_origin);self.grid_cells=np.asarray(grid_cells,int)
            nodes=np.array(list(product(*(range(n+1) for n in self.grid_cells))))*spacing+self.grid_origin
            self.material_grid=MaterialEmbedding(nodes,particle_reference,spacing)
        tree=cKDTree(self.ref)
        self.support=[];self.extension_support=[];self.extension_nodes=0
        for node in self.nodes:
            ids=np.array(tree.query_ball_point(node,spacing*np.sqrt(3)+1.e-10),int)
            w=np.maximum(1-np.abs(self.ref[ids]-node)/spacing,0).prod(axis=1)
            ids,w=ids[w>1.e-12],w[w>1.e-12]
            self.support.append((ids,w))
            # A surface stencil can touch an empty material-grid node on one
            # cut side. Define its one-sided affine continuation from the next
            # reference support ring. This changes visualization only; it does
            # not populate an MPM state or transfer force across a cut.
            ids=np.array(tree.query_ball_point(node,2*spacing*np.sqrt(3)+1.e-10),int)
            w=np.maximum(1-np.abs(self.ref[ids]-node)/(2*spacing),0).prod(axis=1)
            self.extension_support.append((ids[w>1.e-12],w[w>1.e-12]))

    def field(self,x,F,particle_codes,code):
        result=np.full_like(self.nodes,np.nan,dtype=float)
        for i,(ids,w) in enumerate(self.support):
            valid=np.array([compatible(code,c) for c in particle_codes[ids]],dtype=bool)
            ids,w=ids[valid],w[valid]
            if not len(ids):
                ext,ew=self.extension_support[i]
                valid=np.array([compatible(code,c) for c in particle_codes[ext]],dtype=bool)
                ids,w=ext[valid],ew[valid]
                if not len(ids):continue
                self.extension_nodes+=1
            extrapolated=x[ids]+np.einsum('nij,nj->ni',F[ids],self.nodes[i]-self.ref[ids])
            result[i]=(extrapolated*w[:,None]).sum(0)/w.sum()
        return result

    def node_state(self,x,F,particle_codes,node_codes):
        positions=np.full_like(self.nodes,np.nan,dtype=float)
        deformation=np.full((len(self.nodes),3,3),np.nan)
        for i,(ids,w) in enumerate(self.support):
            valid=np.array([compatible(node_codes[i],c) for c in particle_codes[ids]],dtype=bool)
            ids,w=ids[valid],w[valid]
            if not len(ids):continue
            positions[i]=((x[ids]+np.einsum('nij,nj->ni',F[ids],self.nodes[i]-self.ref[ids]))*w[:,None]).sum(0)/w.sum()
            deformation[i]=(F[ids]*w[:,None,None]).sum(0)/w.sum()
        return positions,deformation

    def vertices(self,mesh,x,F,particle_codes):
        if self.material_grid is not None:
            # Surface primal links are padded/staggered for boundary extraction.
            # Embedding must instead use the ORIGINAL MPM grid and its fields.
            fractional=(mesh.reference-self.grid_origin)/self.h
            base=np.minimum(np.maximum(np.floor(fractional).astype(int),0),self.grid_cells-1)
            local=fractional-base
            ids=np.ravel_multi_index((base[:,None,:]+OFFSETS[None,:,:]).transpose(2,0,1),tuple(self.grid_cells+1))
            weights=np.where(OFFSETS[None,:,:]==1,local[:,None,:],1-local[:,None,:]).prod(2)
            embedded=Mesh(mesh.reference,mesh.triangles,mesh.codes,ids,weights)
            return self.material_grid.vertices(embedded,x,F,particle_codes)
        result=np.empty_like(mesh.reference)
        for code in np.unique(mesh.codes):
            field=self.field(x,F,particle_codes,int(code));ids=np.flatnonzero(mesh.codes==code)
            values=field[mesh.cells[ids]];weights=mesh.weights[ids]
            if not np.isfinite(values[weights>1.e-9]).all():
                raise RuntimeError('Surface region has no compatible material support; refine reference grid/support')
            result[ids]=np.nansum(values*weights[...,None],axis=1)
        return result
