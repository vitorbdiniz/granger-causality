import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tools.tools import Bunch, add_constant
from statsmodels.tools.validation import (
    array_like,
    bool_like,
    dict_like,
    float_like,
    int_like,
    string_like,
)

from sklearn.preprocessing import MinMaxScaler

from util import util, padding as pad

import warnings
warnings.filterwarnings("ignore")

"""

    GRANGER CAUSALITY TEST

"""

def granger_causality(characteristics:dict, alfas:dict, fund_filter = None, maxlag=36, statistical_test='params_ftest', separate_lags=True, binary=False,verbose=0):
    '''
        alfas: {dict}   ---     fundo:str -> alfa:pd.DataFrame -> resultados+estatísticas
        characteristics: {dict} --- caracteristica:str -> data:pd.DataFrame -> fundos
    '''
    funds_in_characts = {c:frozenset(characteristics[c].columns) for c in characteristics.keys() }
    funds = fund_filter if fund_filter is not None else alfas.keys()

    results = dict() if separate_lags else pd.DataFrame(columns = funds, index = characteristics.keys())

    i=0
    for fundo in funds:
        i+=1
        if fundo not in alfas.keys():
            continue
        alfa = util.preprocess_serie(alfas[fundo]['alpha'])
        alfa = stationarity_check(alfa, max_iter=5, verbose=0)
        if util.is_none(alfa):
            continue

        if separate_lags:
            results[fundo] = pd.DataFrame(columns=range(1,maxlag+1))
        j=0
        for caracteristica in characteristics.keys():
            j+=1
            pad.verbose(f'{i}.{j}. Testes de Granger --- separate_lags: {separate_lags} --- binary: {binary} --- faltam {len(funds) - i} fundos', level=5, verbose=verbose)
            if fundo not in funds_in_characts[caracteristica]:                
                continue

            selected_characteristic = util.preprocess_serie(characteristics[caracteristica][fundo], dropna=True)
            selected_characteristic = stationarity_check(selected_characteristic, max_iter=5, verbose=0)

            if util.is_none(selected_characteristic):                
                continue

            data = util.join_series(series_list=[alfa, selected_characteristic]).values

            try:
                addconst = int(bool_like(True, "addconst"))
                if data.shape[0] <= 3 * maxlag + addconst:
                    lag = int((data.shape[0] - addconst)/3)-1
                    pad.verbose(f'---- Alterando maxlag para {lag}', level=4, verbose=verbose)
                else:
                    lag = maxlag
                granger_result = granger_causality_test(data, lag, statistical_test, scores=True, separate_lags=separate_lags, binary=binary)
            except:
                continue

            results[fundo].loc[caracteristica] = granger_result
            pad.verbose('--- OK!', level=5, verbose=verbose)
    
    return results

def granger_causality_test(data, maxlag=10, statistical_test='all', scores=True,separate_lags=True, binary=False, verbose=False):
    """
        maxlag : maxlag in range(-1,81)
        statistical_test: {'lrtest', 'params_ftest', 'ssr_chi2test', 'ssr_ftest'}
        scores: {Se True, então retorna o % de testes em que a hipótese foi descartada, senão, retorna um DataFrame com os p-valores}
    """
    if statistical_test == 'all':
        statistical_tests = ['lrtest', 'params_ftest', 'ssr_chi2test', 'ssr_ftest'] 
    elif type(statistical_test) == str:
        statistical_tests = [statistical_test]
    elif type(statistical_test) == list:
        statistical_tests = statistical_test
    else:
        raise AttributeError(f'statistical_test: {statistical_test}')

    gTests = grangercausalitytests(data, maxlag, verbose=verbose)

    result = pd.DataFrame({lag : pd.Series({stat:gTests[lag][0][stat][1] for stat in statistical_tests}) for lag in gTests.keys()}, index = statistical_tests)
    if scores:
        result = stats_scores_per_lag(result, binary=binary) if separate_lags else stats_scores(result, binary=binary)
    return result

def stats_scores(granger_df, binary=False):
    df = granger_df.copy() <= 0.05
    result = 0
    if binary:
        result = 1 if (granger_df <= 0.05).any(1).any() else 0
    else:
        for i in df.index:
            if df.loc[i].any():
                result += 1/df.shape[0]
    return result

def stats_scores_per_lag(granger_df, binary=False):
    df = granger_df.copy() <= 0.05
    Gscores = []
    for col in df.columns:
        Gscore = 0
        if binary:
            Gscore = 1 if df[col].any() else 0
        else:
            for i in df.index:
                if df[col].loc[i]:
                    Gscore += 1/df.shape[0]
        Gscores += [Gscore]
    result = pd.Series(Gscores, index = df.columns)
    return result


"""

    STATIONARITY CHECK

"""


class StationarityError(ValueError):
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


def stationarity_check(serie, max_iter=15, raise_error=False, verbose=0):
    try:
        result = stationarity_transform(serie, max_iter=max_iter, verbose=verbose)
    except:
        result = None
    return result

def stationarity_transform(serie, max_iter=15, verbose=0):
    i = 0
    while max_iter > i and not is_stationarity(serie):
        pad.verbose(f'{i+1}. Não estacionária -> Calculando a {i+1}ª diferença', level=4, verbose=verbose)
        serie = get_difference(serie)
        i += 1
    if not is_stationarity(serie):
        raise StationarityError(f'', f'Série temporal não-estacionária até a {max_iter}ª diferença')
    else:
        if i > 0:
            pad.verbose(f'---- Série estacionária após calcular a {i}ª diferença', level=4, verbose=verbose)
        else:
            pad.verbose(f'---- Série estacionária', level=4, verbose=verbose)
    return serie


def is_stationarity(serie):
    '''
        Teste de Dickey-Fuller aumentado ou Teste ADF identifica se a série possui raíz unitária

        serie: {pd.Series(dtype=np.float)}
        return: {Boolean}
    '''
    return adfuller( serie.dropna() )[1] <= 0.05

def get_difference(serie):
    list_ = pd.Series([None] + [(serie.iloc[i] / serie.iloc[i-1] -1) if serie.iloc[i-1] else 0 for i in range(1, len(serie.index) )], index=serie.index)
    stdev = np.std(list_.dropna() )
    list_ = [ e if e != 0 else stdev/100 for e in list_]

    return pd.Series(list_, serie.index)



"""

    GRANGER SCORES

"""

def granger_scores(granger_results, verbose=0):
    pad.verbose('- Calculando Granger-Scores -', level=3, verbose=verbose)
    if type(granger_results) is dict:
        result = granger_scores_dict(granger_results)
    else:
        result = granger_scores_df(granger_results)
    return result

def granger_scores_df(granger_results):
    result = [granger_results.loc[i].mean() for i in granger_results.index]
    return pd.Series(result, index=granger_results.index)    

def granger_scores_dict(granger_dic):
    """
        granger_dic: dicionário com DataFrames resultantes de granger_tests

    """
    granger_dic = kill_empty_df(granger_dic)
    result = pd.DataFrame(columns = list(granger_dic.values())[0].columns)
    metric_counter = pd.Series(dtype=int)
    for granger_df in granger_dic.values():
        for metric in granger_df.index:
            if metric in metric_counter.index:
                metric_counter[metric] += 1
            else:
                metric_counter[metric] = 1

            if metric in result.index:
                result.loc[metric] += granger_df.loc[metric]
            else:
                result.loc[metric] = granger_df.loc[metric]                
    
    result = pd.DataFrame({lag : result[lag]/metric_counter for lag in result.columns})
    return result
        
def kill_empty_df(dic):
    return {key : dic[key] for key in dic.keys() if dic[key].shape[0] > 0}