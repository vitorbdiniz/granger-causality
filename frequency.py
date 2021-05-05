import pandas as pd
import numpy as np
import datetime as dt

from util import util, padding as pad

def change_frequency(data:pd.DataFrame, nome:str, freq='M', method='sum'):
    df = pd.DataFrame(columns=data.columns, index=data.index)
    for i in range(len(data.columns)):
        print(f'{i}. {nome} -> Freq {freq} -> Faltam {len(data.columns)-i}')
        col = data.columns[i]
        serie = data[col].dropna().resample(freq)
        if method == 'sum':
            serie = serie.sum()
        elif method == 'std':
            serie = serie.std()
        elif method == 'mean':
            serie = serie.mean()
        elif method == 'pad':
            serie = serie.pad()
        elif method == 'cumulative_returns':
            serie = util.retornos_acumulados_por_periodo(data[col], freq)    
        else:
            raise ValueError(f'Método {method} errado')
        df[col] = serie
    return df.dropna(how='all')
def get_method(nome):
    if nome in ('cotistas_economatica','PL_economatica', 'lifetime_economatica'):
        method = 'pad'
    else:
        method = 'sum'
    return method

def generate_data(files_names = None):
    if files_names is None:
        files_names = ['capt_economatica', 'capt_liq_economatica', 'cotistas_economatica', 'PL_economatica', 'resgate_economatica', 'resgate_IR_economatica', 'resgate_total_economatica']
    freq_dic = {'M':'month', 'Q':'quarter', 'Y':'year', 'D':'day'}
    for nome in files_names:
        df = util.df_datetimeindex( pd.read_csv(f'./data/economatica/{nome}.csv', index_col=0) )
        for freq in ['M', 'Q', 'Y']:
            data = change_frequency(df, nome, freq=freq, method=get_method(nome))
            pad.persist_collection({f'{nome}_{freq_dic[freq]}':data}, f'./data/economatica/{freq_dic[freq]}/')
            print(data)

generate_data(['lifetime_economatica'])


