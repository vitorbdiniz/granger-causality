import pandas as pd
import numpy as np

from util import util, padding as pad

def get_results(windows = (12,24,36,48,60,None)):
    granger_results = dict()
    for window in windows:
        pad.verbose(f'Buscando resultados para janela {window}', level=1, verbose=2)
        path = f'./data/granger_tests/month/{window}/maxlags/separated_lags/binary/'
        files = util.get_files(path, except_files=['scores.csv'])
        granger_results[window] = {f.split('.')[0] : pd.read_csv(path+f, index_col=0) for f in files} 
    return granger_results

def all_scores(granger_dic, windows = (12,24,36,48,60,None), maxlags=100, method ='all'):
    scores = dict()
    for window in windows:
        pad.verbose(f'- Calculando Granger-Scores para janela: {window} -', level=3, verbose=5)
        score = granger_scores(granger_dic[window], maxlags=maxlags, method =method)
        pad.persist_collection({'scores':score}, f'./data/granger_tests/compara_scores/{method}/{window}/maxlags/')
        scores[window] = score
    return scores

def granger_scores(granger_dic, maxlags=100, method ='all'):
    """
        granger_dic: dicionário com DataFrames resultantes de granger_tests
    """
    result = pd.DataFrame(columns = [str(i) for i in range(1, maxlags+1)])
    metric_counter = pd.Series(dtype=int)
    for granger_df in granger_dic.values():
        for metric in granger_df.index:
            if metric in metric_counter.index:
                metric_counter[metric] += 1
            else:
                metric_counter[metric] = 1

            gScore = pd.Series(granger_df.loc[metric], index = result.columns).fillna(0)
            if metric in result.index:
                result.loc[metric] += gScore
            else:
                result.loc[metric] = gScore

    result = result/len(granger_dic) if method == 'all' else pd.DataFrame({lag : result[lag]/metric_counter for lag in result.columns})
    return result
        
def kill_empty_df(dic):
    return {key : dic[key] for key in dic.keys() if dic[key].shape[0] > 0}


results = get_results()
for method in ['all', 'selected']:
    all_scores(results, method=method)
print(results)

