import math, random, statistics, itertools

SEED = 20261027

def quantile(xs, p):
    ys = sorted(xs); pos = p*(len(ys)-1); lo=int(pos); hi=min(lo+1,len(ys)-1)
    return ys[lo]+(pos-lo)*(ys[hi]-ys[lo])

def build():
    rng = random.Random(SEED)
    obs=[]
    while len(obs)<512:
        z=rng.gauss(0,1); d=24.6+1.95*z+0.22*(z*z-1)
        if not (19.5<=d<=31.5): continue
        c=0.30+0.045*(d-24.0)+rng.gauss(0,0.21)
        obs.append((round(d,2),min(0.97,max(0.02,c))))
    sim=[]
    while len(sim)<1000:
        b=rng.uniform(-2,2); d=24.5+1.42*b+0.10*(b*b-1.33)
        if not (19.5<=d<=31.5): continue
        c=0.30+0.045*(d-24.0)+rng.gauss(0,0.15)
        sim.append((round(d,2),min(0.72,max(0.02,c))))
    return obs,sim

def make_prob(sizes, target, klo, khi, kcalc, b0):
    def p(d,c):
        o=min(((s-d)/d for s in sizes), key=lambda x: abs(x-target))
        if o<0.0 or o>0.45: return 0.0
        pen = klo*(target-o)**2 if o<target else khi*(o-target)**2
        return 1/(1+math.exp(-(b0-pen-kcalc*c)))
    return p

def stats(p, obs, sim):
    ps=sum(p(d,c) for d,c in sim)/len(sim)
    po=sum(p(d,c) for d,c in obs)/len(obs)
    ys=sorted(obs,key=lambda t:t[0]); step=len(ys)//10
    dec=[]
    for i in range(10):
        ch=ys[i*step:(i+1)*step] if i<9 else ys[9*step:]
        dec.append(sum(p(d,c) for d,c in ch)/len(ch))
    return ps,po,ps-po,dec

obs,sim=build()
print("obs n",len(obs),"sim n",len(sim))
print("obs D p99",round(quantile([d for d,_ in obs],0.99),2),"sim D p99",round(quantile([d for d,_ in sim],0.99),2))
print("obs C p99",round(quantile([c for _,c in obs],0.99),3),"max",round(max(c for _,c in obs),3))
print("obs frac C>0.72",round(sum(1 for _,c in obs if c>0.72)/len(obs),3))

for sizes in [(20,22,24,26,28,30),(20,23,26,29)]:
    for target in (0.12,0.15):
        for klo in (700,1200):
            p=make_prob(sizes,target,klo,20,2.2,3.4)
            ps,po,diff,dec=stats(p,obs,sim)
            print(f"sizes{sizes} t={target} klo={klo} ps={ps:.4f} po={po:.4f} opt={diff:+.4f} "
                  f"d1={dec[0]:.3f} d10={dec[-1]:.3f} d5={dec[4]:.3f}")
