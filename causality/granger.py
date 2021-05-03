import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from sklearn.preprocessing import MinMaxScaler

from util import util, padding as pad


"""

    GRANGER CAUSALITY TEST

"""

def granger_causality(characteristics:dict, alfas:dict, maxlag=15, statistical_test='params_ftest', verbose=0):
    '''
        alfas: {dict}   ---     fundo:str -> alfas:pd.DataFrame -> resultados+estatísticas
        characteristics: {dict} --- caracteristica:str -> data:pd.DataFrame -> fundos
    '''
    results = dict()
    i=0
    for fundo in alfas.keys():
        i+=1
        alfa = util.preprocess_serie(alfas[fundo]['alpha'])
        alfa = stationarity_check(alfa, max_iter=5, verbose=verbose)
        if util.is_none(alfa):
            continue
        
        results[fundo] = pd.DataFrame(columns=range(1,maxlag+1))
        j=0
        for caracteristica in characteristics.keys():
            j+=1
            pad.verbose(f'{i}.{j}. Testes de Granger --- faltam {len(alfas.keys()) - i} fundos --- faltam {len(characteristics.keys()) - j} caracteristicas', level=5, verbose=verbose)

            selected_characteristic = util.preprocess_serie(characteristics[caracteristica][fundo])
            selected_characteristic = stationarity_check(selected_characteristic, max_iter=5, verbose=verbose)
            if util.is_none(selected_characteristic):
                continue
            data = util.join_series(series_list=[alfa, selected_characteristic]).values
            granger_result = granger_causality_test(data, maxlag, statistical_test, scores=True)
            results[fundo].loc[caracteristica] = granger_result
    
    return results

def granger_causality_test(data, maxlag=10, statistical_test='all', scores=True,verbose=False):
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
        result = granger_stats_scores(result)
    return result

def granger_stats_scores(granger_df):
    df = granger_df.copy() <= 0.05
    Gscores = []
    for col in df.columns:
        Gscore = 0
        for i in df.index:
            if df[col].loc[i]:
                Gscore += 1/df.shape[0]
        Gscores += [Gscore]
    return pd.Series(Gscores, index = df.columns)



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

def granger_scores(granger_dic:dict ):
    """
        granger_dic: dicionário com DataFrames resultantes de granger_tests

    """
    result = pd.DataFrame(columns = list(granger_dic.values())[0].columns)

    for granger_df in granger_dic.values():
        for metric in granger_df.index:
            if metric in result.index:
                result.loc[metric] += granger_df.loc[metric]
            else:
                result.loc[metric] = granger_df.loc[metric]                
    result /= len(granger_dic.values())
    return result
        
