"""Find a K-regular integer matrix whose (N,C) decomposition space has a
NON-TRIVIAL Pareto frontier: at least two non-dominated points, so that the
DCT-optimal decomposition CHANGES as T_config is swept (small T_config favours
min-C strong/BvN; large T_config favours min-N). Equivalently: min-N decomp has
C > S, and min-C (strong) decomp has N > min-N.
"""
import numpy as np, math

def rpm(n,rng):
    p=rng.permutation(n);M=np.zeros((n,n),int);M[np.arange(n),p]=1;return M
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
def ocol(M,k):
    if k==0:return 1 if not M.any() else 0
    c=0
    for e in perfect_matchings(M):
        if me(M,e)>=1:c+=ocol(M-mat_of(e,M.shape[0]),k-1)
    return c
def enum(M,cache,cap):
    key=M.tobytes()
    if key in cache:return cache[key]
    if not M.any():cache[key]={frozenset()};return cache[key]
    res=set()
    for e in all_matchings(M):
        Mm=mat_of(e,M.shape[0]);m=me(M,e);ek=tuple(sorted(e))
        for a in range(1,m+1):
            for s in enum(M-a*Mm,cache,cap):
                d=dict(s);d[ek]=d.get(ek,0)+a;res.add(frozenset(d.items()))
                if len(res)>cap:cache[key]=res;return res
    cache[key]=res;return res

def pareto(points):
    # points: set of (N,C). Non-dominated (minimize both).
    pts=sorted(set(points))
    nd=[]
    for N,C in pts:
        if not any(n2<=N and c2<=C and (n2,c2)!=(N,C) for n2,c2 in pts):
            nd.append((N,C))
    return nd

# candidate generator with cheap filters, then full eval, then pareto test
for n,K in [(4,3),(4,4),(5,3)]:
    rng=np.random.default_rng(99)
    seen=set();tested=0
    for t in range(40000):
        D=sum(rpm(n,rng) for _ in range(K))
        if not ((D.sum(0)==K).all() and (D.sum(1)==K).all()):continue
        mk=D.tobytes()
        if mk in seen:continue
        seen.add(mk)
        if D.max()<2:continue
        oc=ocol(D,K)
        if oc==0 or oc%math.factorial(K)!=0:continue
        uns=oc//math.factorial(K)
        if uns<1:continue
        tested+=1
        cache={}
        ds=[dict(s) for s in enum(D,cache,30000)]
        if len(ds)>25000:continue
        pts={(len(d),sum(d.values())) for d in ds}
        nd=pareto(pts)
        # want a non-trivial frontier: min-N point has C>K (strong), i.e.
        # the (minN) and (minC=K) points differ.
        minN=min(N for N,C in nd)
        cAtMinN=min(C for N,C in nd if N==minN)
        minC=min(C for N,C in nd)
        nAtMinC=min(N for N,C in nd if C==minC)
        if len(nd)>=2 and cAtMinN>minC and nAtMinC>minN:
            print(f"FOUND n={n} K={K} strong={uns} total={len(ds)} frontier={nd}")
            print(f"  minN point=({minN},{cAtMinN})  minC point=({nAtMinC},{minC})")
            print(repr(D));print()
            break
    print(f"(n={n} K={K}: tested {tested} distinct matrices)")
