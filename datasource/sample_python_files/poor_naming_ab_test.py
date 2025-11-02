import pandas as pd
import numpy as np
from scipy import stats

def f(x):
    y=pd.read_csv(x)
    return y

def g(a,b,c=0.05):
    d,e=stats.ttest_ind(a,b)
    f=np.mean(a)
    g=np.mean(b)
    h=np.sqrt((np.var(a,ddof=1)+np.var(b,ddof=1))/2)
    i=(g-f)/h if h>0 else 0
    return {'s':d,'p':e,'sig':e<c,'m1':f,'m2':g,'es':i}

def h(x,y,z,w,a=0.05):
    t1=np.array([[x,y-x],[z,w-z]])
    c,p,_,_=stats.chi2_contingency(t1)
    r1=x/y
    r2=z/w
    u=((r2-r1)/r1*100) if r1>0 else 0
    return {'c':c,'p':p,'sig':p<a,'r1':r1,'r2':r2,'u':u}

def j(d,m,a=0.05):
    c=d[d['group']=='control'][m].values
    t=d[d['group']=='treatment'][m].values
    r=g(c,t,a)
    r['n1']=len(c)
    r['n2']=len(t)
    return r

def main():
    p="datasource/data/sample_ab_test.csv"
    d=f(p)
    print("loaded")
    print(f"n: {len(d)}")

    r1=j(d,'revenue')
    print(f"m1: ${r1['m1']:.2f}")
    print(f"m2: ${r1['m2']:.2f}")
    print(f"p: {r1['p']:.4f}")

    x1=d[d['group']=='control']['conversion'].sum()
    y1=len(d[d['group']=='control'])
    x2=d[d['group']=='treatment']['conversion'].sum()
    y2=len(d[d['group']=='treatment'])

    r2=h(x1,y1,x2,y2)
    print(f"r1: {r2['r1']:.2%}")
    print(f"r2: {r2['r2']:.2%}")
    print(f"u: {r2['u']:.2f}%")

if __name__=="__main__":
    main()
