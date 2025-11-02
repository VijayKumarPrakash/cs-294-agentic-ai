import pandas as pd
import numpy as np
from scipy import stats

def load_data(file_path):
  data=pd.read_csv(file_path)
  return data

def ttest_analysis(control,treatment,alpha=0.05):
  t_stat,p_val=stats.ttest_ind(control,treatment)
  c_mean=np.mean(control)
  t_mean=np.mean(treatment)
  pooled_std=np.sqrt((np.var(control,ddof=1)+np.var(treatment,ddof=1))/2)
  effect=(t_mean-c_mean)/pooled_std if pooled_std>0 else 0
  return {'t_stat':t_stat,'p_val':p_val,'sig':p_val<alpha,'c_mean':c_mean,'t_mean':t_mean,'effect':effect}

def chi_square_test(c_conv,c_total,t_conv,t_total,alpha=0.05):
    table=np.array([[c_conv,c_total-c_conv],[t_conv,t_total-t_conv]])
    chi2,p_val,_,_=stats.chi2_contingency(table)
    c_rate=c_conv/c_total
    t_rate=t_conv/t_total
    uplift=((t_rate-c_rate)/c_rate)*100 if c_rate>0 else 0
    return {'chi2':chi2,'p_val':p_val,'sig':p_val<alpha,'c_rate':c_rate,'t_rate':t_rate,'uplift':uplift}

def analyze_metric(data,metric,alpha=0.05):
    control=data[data['group']=='control'][metric].values
    treatment=data[data['group']=='treatment'][metric].values
    results=ttest_analysis(control,treatment,alpha)
    results['n_control']=len(control)
    results['n_treatment']=len(treatment)
    return results

def main():
  path="datasource/data/sample_ab_test.csv"
  df=load_data(path)
  print("loaded data")
  print(f"total: {len(df)}")

  print("\nrevenue analysis")
  rev_res=analyze_metric(df,'revenue')
  print(f"control: ${rev_res['c_mean']:.2f}")
  print(f"treatment: ${rev_res['t_mean']:.2f}")
  print(f"effect: {rev_res['effect']:.3f}")
  print(f"p-value: {rev_res['p_val']:.4f}")

  print("\nconversion analysis")
  c_conv=df[df['group']=='control']['conversion'].sum()
  c_tot=len(df[df['group']=='control'])
  t_conv=df[df['group']=='treatment']['conversion'].sum()
  t_tot=len(df[df['group']=='treatment'])
  conv_res=chi_square_test(c_conv,c_tot,t_conv,t_tot)
  print(f"control rate: {conv_res['c_rate']:.2%}")
  print(f"treatment rate: {conv_res['t_rate']:.2%}")
  print(f"uplift: {conv_res['uplift']:.2f}%")

if __name__=="__main__":
  main()
