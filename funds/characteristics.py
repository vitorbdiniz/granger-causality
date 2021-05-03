import pandas as pd
import numpy as np
import statistics as st
import datetime as dt

from factors.factors import nefin_risk_free, nefin_single_factor
from util import util,padding as pad

def extract_characteristics(alphas:dict, fias_characteristics = None, freq = 'M', window = None, dropna="all", verbose=0):
    '''
        Calcula características do dict `fias_characteristics` que contém informações dos fundos.
        alphas: {dict} -> contendo o alfa e os betas em dict de DataFrames;
        fias_characteristics: {dict, None} -> se None, então a busca é automática
        window: {'M', 'Q', 'Y'}

        return: {dict} -> dict de DataFrames de fundos. Cada DF é uma característica específica
    '''
    if fias_characteristics is None:
        fias_characteristics = get_characts(freq)

    result = {
        'variation'          : get_return_df(fias_characteristics['fis_acc'], window=window),
        'sharpe'             : get_sharpe_ratio_df(fias_characteristics['fis_acc'], freq=freq, window=window, verbose=verbose),
        'treynor'            : get_treynor_ratio_df(fias_characteristics['fis_acc'], alphas, freq=freq, window=window, verbose=verbose),
        'lifetime'           : fias_characteristics['lifetime'],
        'information_ratio'  : information_ratio_df(fias_characteristics['fis_acc'] , freq=freq, window=window, verbose=verbose),
        'standard_deviation' : volatility_df(fias_characteristics['fis_acc'], method='std', window=window, verbose=verbose),
        'downside_deviation' : volatility_df(fias_characteristics['fis_acc'], method='dsd', window=window, verbose=verbose),
        'equity'             : fias_characteristics['PL'],
        'captacao'           : fias_characteristics['capt'],
        'captacao_liquida'   : fias_characteristics['capt_liq'],
        'cotistas'           : fias_characteristics['cotistas']
    }
    return result

def get_characts(freq='M'):
    freq_dic = {'M':'month', 'Q':'quarter', 'Y':'year', 'D':'day'}
    names = ['fis_acc', 'capt', 'capt_liq', 'cotistas', 'PL', 'resgate', 'resgate_IR', 'resgate_total', 'lifetime']
    characteristics = {n : util.df_datetimeindex(pd.read_csv(f'./data/economatica/{freq_dic[freq]}/{n}_economatica_{freq_dic[freq]}.csv', index_col=0)) for n in names}
    return characteristics

def capm_risk_premium(returns, Rf=None, freq = None, window=None):
    if Rf is None:
        Rf = nefin_risk_free(freq=freq)
    Rp = (returns - nefin_risk_free()).dropna()
    risk_premium = get_return(Rp, window=window)
    return risk_premium

'''
    SHARPE
'''
def get_sharpe_ratio(returns:pd.Series, Rf=None, freq=None, window=None):
    risk_premium = capm_risk_premium(returns, Rf=Rf, freq=freq, window=window)
    vol = volatility(returns,window=window)
    sharpe = ( risk_premium / vol ).dropna()
    return sharpe

def get_sharpe_ratio_df(returns:pd.DataFrame, freq=freq, window=None, verbose=0):
    result = pd.DataFrame(columns = returns.columns, index = returns.index)
    Rf = nefin_risk_free(freq=freq)
    for i in returns.shape[1]:
        portfolio = returns.columns[i]
        pad.verbose(f'{i}. Índice de Sharpe --- frequência {freq} --- janela {window} --- Faltam {returns.shape[1]-i}', level=5, verbose=verbose)
        result[portfolio] = get_sharpe_ratio(returns[portfolio], Rf=Rf, freq=freq, window=window)
    return result.dropna(how='all')

'''
    TREYNOR
'''

def get_treynor_ratio(returns:pd.Series, betas:pd.Series, Rf:pd.Series=None, freq=None, window=None):
    risk_premium = capm_risk_premium(returns, Rf=Rf, freq=freq, window=window)
    treynor = ( risk_premium / betas ).dropna()
    return treynor


def get_treynor_ratio_df(returns:pd.DataFrame, alphas:dict, freq=None, window=None, verbose=0):
    result = pd.DataFrame(columns = returns.columns, index = returns.index)
    Rf = nefin_risk_free(freq=freq)
    for i in returns.shape[1]:
        portfolio = returns.columns[i]
        pad.verbose(f'{i}. Índice de Treynor --- frequência {freq} --- janela {window} --- Faltam {returns.shape[1]-i}', level=5, verbose=verbose)
        result[portfolio] = get_treynor_ratio(returns[portfolio], alphas[portfolio]['beta_Market'], Rf=Rf, freq=freq, window=window)
    return result.dropna(how='all')

'''
    VOLATILITY
'''
 
def volatility_df(returns:pd.DataFrame, method="std", window=None, verbose=0):
    result = pd.DataFrame(columns = returns.columns, index = returns.index)
    for i in returns.shape[1]:
        portfolio = returns.columns[i]
        pad.verbose(f'{i}. Volatilidade --- método {method} --- frequência {freq} --- janela {window} --- Faltam {returns.shape[1]-i}', level=5, verbose=verbose)
        result[portfolio] = volatility(returns[portfolio], method=method, window=window)
    return result.dropna(how='all')

def volatility(returns:pd.Series, method="std", window=None):
    """
        Calculate volatility by the sample's standard deviation(std) or by its downside deviation (dsd)
        method: {'std', 'dsd'}
        cumulative: se cumulative < 0, então é calculada a volatilidade acumulada, senão, é calculada dentro da janela de `cumulative` meses
        returns: pd.Series(dtype=np.float)
    """
    if len(returns) < 2:
        return None

    if window is None:
        vol = cumulative_volatility(returns, method)
    else:
        window *= 22
        vol = window_volatility(returns, method, window)
    return vol

def window_volatility(returns, method, window):
    if method == "std":
        vol = pd.Series( [(returns.iloc[i-window : i]).std() for i in range(window, returns.shape[0]+1)], index = returns.index [window-1:len(returns)])
    elif method == "dsd":
        vol = pd.Series([downside_deviation(returns.iloc[i-window : i]) for i in range(window, returns.shape[0]+1)], index = returns.index[window-1:len(returns)])
    return vol

def cumulative_volatility(returns, method="std"):
    if method == "std":
        vol = pd.Series( [(returns.iloc[0:i]).std() for i in range(1, returns.shape[0]+1)], index = returns.index )
    elif method == "dsd":
        vol = pd.Series([downside_deviation(returns.iloc[0:i]) for i in range(1, returns.shape[0]+1)], index = returns.index)
    return vol

def downside_deviation(returns):
    avg = st.mean(returns.values)
    down_values = returns[returns < avg]
    if len(down_values) < 2:
        return None
    return st.stdev(down_values, xbar=avg)


'''
    INFORMATION RATIO
'''
def information_ratio_df(returns:pd.DataFrame, freq=None, window=None, verbose=0):
    result = pd.DataFrame(columns = returns.columns, index = returns.index)
    MKT = nefin_single_factor(factor="Market", freq=freq)
    for i in returns.shape[1]:
        portfolio = returns.columns[i]
        pad.verbose(f'{i}. Information-Ratio --- janela {window} --- Faltam {returns.shape[1]-i}', level=5, verbose=verbose)
        result[portfolio] = information_ratio(returns[portfolio], MKT, window=window)
    return result.dropna(how='all')

def information_ratio(returns:pd.Series, mkt_returns=None, window=None):
    difference = (returns - MKT).dropna()
    premium = get_return(difference, window=window)
    tracking_error = volatility( difference, method='std', window=window )
    IR = premium / tracking_error
    return IR.dropna()


'''
    RETORNOS ACUMULADOS POR JANELA
'''

def get_return(retornos, window=None, return_type = pd.Series):
    '''
        Calcula retornos acumulados dentro de uma janela móvel. Se window is None, então a janela é do tamanho da amostra 
    '''
    if window is None:
        retornos_acc = util.cumulative_return(retornos, return_type=return_type)
    else:
        retornos_acc = [util.cumulative_return(retornos[i-window:i], return_type=list)[-1] for i in range(window, len(retornos)+1)]
        retornos_acc = pd.Series(retornos_acc, index=retornos.index[window-1:len(retornos)])
    return retornos_acc

def get_return_df(returns:pd.DataFrame, window=None):
    for i in returns.shape[1]:
        portfolio = returns.columns[i]
        pad.verbose(f'{i}. Retorno acumulado --- janela {window} --- Faltam {returns.shape[1]-i}', level=5, verbose=verbose)
        result[portfolio] = get_return(returns[portfolio], window=window)
    return result.dropna(how='all')


