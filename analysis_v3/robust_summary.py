import pandas as pd, glob
rows=[]
for f in sorted(g for g in glob.glob('plant_sens_*.csv') if 'summary' not in g):
    d=pd.read_csv(f); sc=d.scen.iloc[0]; k=d.k.iloc[0]; Kp=k*17.5/100
    d['cs']=d.fuel-Kp*(d.socf-50)
    g=d.groupby('ctrl').cs.mean(); t=(~d[d.ctrl=='G/MA'].term.astype(str).isin(['None','nan'])).sum() if 'term' in d else 0
    rows.append(dict(scen=sc,k=k,ECMS=g['A-ECMS'],RB=g['RB'],GMA=g['G/MA'],gap_GMA=100*(g['G/MA']/g['A-ECMS']-1),gap_RB=100*(g['RB']/g['A-ECMS']-1),
                     par_ecms=d[d.ctrl=='A-ECMS'].par.mean(),par_gma=d[d.ctrl=='G/MA'].par.mean(),ser_gma=d[d.ctrl=='G/MA'].ser.mean(),term=t))
r=pd.DataFrame(rows).set_index('scen'); print(r.round(1)); r.to_csv('plant_sens_summary.csv')
