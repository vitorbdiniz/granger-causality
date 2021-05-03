import pandas as pd
import datetime as dt
from util import util, padding as pad

def join_dfs(nome, since = None, until = None, persist = False):
    cotas = util.df_datetimeindex( pd.read_csv(f'/home/vitorbdiniz/Downloads/{nome}.csv', index_col=0   , na_values=['-', '']) )
    cotas1 = util.df_datetimeindex(pd.read_csv(f'/home/vitorbdiniz/Downloads/{nome}(1).csv', index_col=0, na_values=['-', '']) )
    cotas2 = util.df_datetimeindex(pd.read_csv(f'/home/vitorbdiniz/Downloads/{nome}(2).csv', index_col=0, na_values=['-', '']) )

    joined = cotas.join(cotas1, how='outer')
    joined = joined.join(cotas2, how='outer')
    
    if since is not None and until is not None:
        start = joined.index.get_loc( since, method='nearest') if since is not None else 0    
        end = joined.index.get_loc( until, method='pad') if until is not None else joined.shape[0]+1
        joined = joined.iloc[start:end].copy()

    pad.persist(joined, f'./data/{nome}_economatica.csv', to_persist=persist)

    return joined


def to_int(string:str):
    string = string.replace('.', '')
    return int(string)

def lifetime():
    df = join_dfs('capt_liq')
    df = util.drop_duplicate_index(df)
    result = pd.DataFrame(columns=df.columns, index = df.index)
    for col in df.columns:
        print(col)
        serie = df[col].dropna()
        serie = pd.Series(range(1, serie.shape[0]+1), index = serie.index)
        result[col] = serie
    result.to_csv(f'./data/lifetime_economatica.csv')
    return result

print(lifetime())