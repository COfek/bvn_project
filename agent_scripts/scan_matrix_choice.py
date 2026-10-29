"""Find a K-regular integer matrix that simultaneously gives:
  (i)  several STRONG decompositions = unordered K-edge-colorings (clean: the
       ordered-coloring count divides by K! to a positive integer that matches
       the brute-force count of coeff-1 perfect-matching partitions);
  (ii) a rich MATCHING-based decomposition space with C-spread (C from S up);
  (iii) total enumerable (a few hundred / few thousand).
"""
import numpy as np, itertools, math

def rpm(n, rng):
    p=rng.permutation(n); M=np.zeros((n,n),int); M[np.arange(n),p]=1; return M

def perfect_matchings(M):
    n=M.shape[0]; out=[]
    def rec(r,used,cur):
        if r==n: out.append(tuple(cur)); return
        for c in range(n):
            if not used[c] and M[r,c]>0:
                used[c]=True; cur.append((r,c)); rec(r+1,used,cur); cur.pop(); used[c]=False
    rec(0,[False]*n,[]); return out

def all_matchings(M):
    n=M.shape[0]; cells=[(i,j) for i in range(n) for j in range(n) if M[i,j]>0]; res=[]
    def rec(s,ur,uc,cur):
        if cur: res.append(tuple(cur))
        for k in range(s,len(cells)):
            i,j=cells[k]
            if i not in ur and j not in uc:
                ur.add(i);uc.add(j);cur.append((i,j));rec(k+1,ur,uc,cur);cur.pop();ur.discard(i);uc.discard(j)
    rec(0,set(),set(),[]); return res

def mat_of(e,n):
    M=np.zeros((n,n),int)
    for i,j in e: M[i,j]=1
    return M

def minedge(M,e): return min(M[i,j] for i,j in e)

def ordered_colorings(M,k):
    if k==0: return 1 if not M.any() else 0
    c=0
    for e in perfect_matchings(M):
        if minedge(M,e)>=1: c+=ordered_colorings(M-mat_of(e,M.shape[0]),k-1)
    return c

def enum_match(M,cache,cap):
    key=M.tobytes()
    if key in cache: return cache[key]
    if not M.any(): cache[key]={frozenset()}; return cache[key]
    res=set()
    for e in all_matchings(M):
        Mm=mat_of(e,M.shape[0]); m=minedge(M,e); ek=tuple(sorted(e))
        for a in range(1,m+1):
            for s in enum_match(M-a*Mm,cache,cap):
                d=dict(s); d[ek]=d.get(ek,0)+a; res.add(frozenset(d.items()))
                if len(res)>cap: cache[key]=res; return res
    cache[key]=res; return res

def strong_count(M,K):
    # coeff-1 perfect-matching partitions, unordered
    cache={}
    cnt=0
    ds=enum_match(M,cache,10**9) if False else None
    # direct: ordered/K!
    return None

found=[]
for n,K in [(4,3)]:
    rng=np.random.default_rng(2024)
    seen_mat=set()
    for t in range(20000):
        D=sum(rpm(n,rng) for _ in range(K))
        if not ((D.sum(0)==K).all() and (D.sum(1)==K).all()): continue
        mk=D.tobytes()
        if mk in seen_mat: continue
        seen_mat.add(mk)
        if D.max()<2: continue                  # need entries>=2 for matching C-spread
        oc=ordered_colorings(D,K)
        if oc==0: continue
        if oc % math.factorial(K)!=0: continue
        uns=oc//math.factorial(K)
        if uns<2: continue                      # want >=2 strong decompositions
        found.append((n,K,uns,D.copy()))
        if len(found)>=12: break
    if len(found)>=12: break

# Evaluate the cheap-filtered candidates fully, pick best C-spread.
results=[]
for n,K,uns,D in found:
    cache={}
    ds=[dict(s) for s in enum_match(D,cache,200000)]
    Ns=[len(d) for d in ds]; Cs=[sum(d.values()) for d in ds]
    results.append((max(Cs)-min(Cs), len(ds), uns, min(Ns),max(Ns),min(Cs),max(Cs), n,K, D))
results.sort(key=lambda r: (r[0], r[1]), reverse=True)
for cspread,tot,uns,nlo,nhi,clo,chi,n,K,D in results[:6]:
    print(f"n={n} K={K} strong={uns} total={tot} N={nlo}..{nhi} C={clo}..{chi} Cspread={cspread}")
    print(repr(D)); print()
