"""Compute the (N,C) Pareto frontier of the chosen matrix's full decomposition
space and check whether the DCT argmin can move with T_config."""
import numpy as np

def perfect_matchings(M):
    n=M.shape[0];out=[]
    def rec(r,u,c):
        if r==n:out.append(tuple(c));return
        for j in range(n):
            if not u[j] and M[r,j]>0:
                u[j]=True;c.append((r,j));rec(r+1,u,c);c.pop();u[j]=False
    rec(0,[False]*n,[]);return out
def all_matchings(M):
    n=M.shape[0];cells=[(i,j) for i in range(n) for j in range(n) if M[i,j]>0];res=[]
    def rec(s,ur,uc,cur):
        if cur:res.append(tuple(cur))
        for k in range(s,len(cells)):
            i,j=cells[k]
            if i not in ur and j not in uc:
                ur.add(i);uc.add(j);cur.append((i,j));rec(k+1,ur,uc,cur);cur.pop();ur.discard(i);uc.discard(j)
    rec(0,set(),set(),[]);return res
def mat_of(e,n):
    M=np.zeros((n,n),int)
    for i,j in e:M[i,j]=1
    return M
def me(M,e):return min(M[i,j] for i,j in e)
def enum(M,cache):
    key=M.tobytes()
    if key in cache:return cache[key]
    if not M.any():cache[key]={frozenset()};return cache[key]
    res=set()
    for e in all_matchings(M):
        Mm=mat_of(e,M.shape[0]);m=me(M,e);ek=tuple(sorted(e))
        for a in range(1,m+1):
            for s in enum(M-a*Mm,cache):
                d=dict(s);d[ek]=d.get(ek,0)+a;res.add(frozenset(d.items()))
    cache[key]=res;return res

# candidates to examine (all K=3-regular n=4)
cands={
 "A":np.array([[1,1,0,1],[1,1,1,0],[1,1,1,0],[0,0,1,2]]),
 "B":np.array([[0,2,1,0],[1,0,1,1],[1,1,0,1],[1,0,1,1]]),
 "C":np.array([[0,2,0,1],[0,1,1,1],[1,0,1,1],[2,0,1,0]]),
 "D":np.array([[1,1,0,1],[1,0,0,2],[1,1,1,0],[0,1,2,0]]),
}
for name,D in cands.items():
    cache={}
    ds=[dict(s) for s in enum(D,cache)]
    pts={}
    for d in ds:
        N=len(d);C=sum(d.values())
        pts.setdefault((N,C),0); pts[(N,C)]+=1
    allp=sorted(pts)
    nd=[(N,C) for N,C in allp if not any(n2<=N and c2<=C and (n2,c2)!=(N,C) for n2,c2 in allp)]
    minN=min(N for N,C in nd); cAtMinN=min(C for N,C in nd if N==minN)
    minC=min(C for N,C in nd); nAtMinC=min(N for N,C in nd if C==minC)
    moves = (cAtMinN>minC and nAtMinC>minN)
    print(f"{name}: total={len(ds)} frontier(N,C)={nd}")
    print(f"   minN=({minN},{cAtMinN}) minC=({nAtMinC},{minC}) -> optimum_moves={moves}")
