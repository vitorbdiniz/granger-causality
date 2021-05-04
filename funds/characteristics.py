import pandas as pd
import numpy as np
import statistics as st
import datetime as dt

from factors.factors import nefin_risk_free, nefin_single_factor
from util import util,padding as pad

def extract_characteristics(fias_characteristics = None, freq = 'M', window = None, verbose=0):
    '''
        Calcula características do dict `fias_characteristics` que contém informações dos fundos.
        alphas: {dict} -> contendo o alfa e os betas em dict de DataFrames;
        fias_characteristics: {dict, None} -> se None, então a busca é automática
        window: {'M', 'Q', 'Y'}

        return: {dict} -> dict de DataFrames de fundos. Cada DF é uma característica específica
    '''

    windows_dict = {'D': 22, 'M':1, 'Q':0.25, 'Y':1/12}
    w = int(windows_dict[freq] * window) if window is not None else window

    if fias_characteristics is None:
        fias_characteristics = get_characts(freq)
    #std = volatility_df(fias_characteristics['fis_acc'], method='std', window=window, verbose=verbose)

    result = {
        #'variation'          : get_return_df(fias_characteristics['fis_acc'], window=w, verbose= verbose),
        #'sharpe'             : get_sharpe_ratio_df(fias_characteristics['fis_acc'],  std, freq=freq, window=w, verbose=verbose),
        #'treynor'            : get_treynor_ratio_df(fias_characteristics['fis_acc'], betas=window,freq=freq, window=w, verbose=verbose),
        #'lifetime'           : fias_characteristics['lifetime'],
        #'information_ratio'  : information_ratio_df(fias_characteristics['fis_acc'] , freq=freq, window=w, verbose=verbose),
        #'standard_deviation' : std,
        #'downside_deviation' : volatility_df(fias_characteristics['fis_acc'], method='dsd', window=w, verbose=verbose),
        #'equity'             : fias_characteristics['PL'],
        #'cotistas'           : fias_characteristics['cotistas'],
        'captacao'           : trailing_sum_df(fias_characteristics['capt'], window =w, verbose=verbose),
        'captacao_liquida'   : trailing_sum_df(fias_characteristics['capt_liq'],window =w, verbose=verbose),
        'resgate'            : trailing_sum_df(fias_characteristics['resgate'], window =w, verbose=verbose),
        'resgate_IR'         : trailing_sum_df(fias_characteristics['resgate_IR'], window =w, verbose=verbose),
        'resgate_total'      : trailing_sum_df(fias_characteristics['resgate_total'], window =w, verbose=verbose)
    }
    return result

def get_characts(freq='M'):
    freq_dic = {'M':'month', 'Q':'quarter', 'Y':'year', 'D':'day'}
    names = ['fis_acc', 'capt', 'capt_liq', 'cotistas', 'PL', 'resgate', 'resgate_IR', 'resgate_total', 'lifetime']
    characteristics = {n : util.df_datetimeindex(pd.read_csv(f'./data/economatica/{freq_dic[freq]}/{n}_economatica_{freq_dic[freq]}.csv', index_col=0)) for n in names}
    characteristics = {name : util.reindex_timeseries(characteristics[name], freq=freq) for name in characteristics.keys()}
    return characteristics

'''
    PRÊMIO PELO RISCO DE MERCADO
'''

def capm_risk_premium(returns, Rf=None, freq = None, window=None):
    if Rf is None:
        Rf = nefin_risk_free(freq=freq)
    Rp = (returns - Rf).dropna()
    risk_premium = get_return(Rp, window=window)
    return risk_premium

'''
    SHARPE
'''
def get_sharpe_ratio(returns:pd.Series, vol = None, Rf=None, freq=None, window=None):
    if vol is None:
        vol = volatility(returns,window=window)
    risk_premium = capm_risk_premium(returns, Rf=Rf, freq=freq, window=window)
    sharpe = ( risk_premium / vol ).dropna()
    return sharpe

def get_sharpe_ratio_df(returns:pd.DataFrame, standard_deviations = None, freq=None, window=None, verbose=0):
    pad.verbose('- Sharpe-ratio -', level=2, verbose=verbose)
    result = pd.DataFrame(columns = returns.columns, index = returns.index)
    Rf = nefin_risk_free(freq=freq)
    for i in range(returns.shape[1]):
        portfolio = returns.columns[i]
        pad.verbose(f'{i}. Índice de Sharpe --- frequência {freq} --- janela {window} --- Faltam {returns.shape[1]-i}', level=5, verbose=verbose)
        result[portfolio] = get_sharpe_ratio(returns[portfolio], standard_deviations[portfolio], Rf=Rf, freq=freq, window=window)

    pad.verbose('line', level=2, verbose=verbose)
    return result.dropna(how='all')

'''
    TREYNOR
'''

def get_treynor_ratio(returns:pd.Series, betas:pd.Series, Rf:pd.Series=None, freq=None, window=None):
    risk_premium = capm_risk_premium(returns, Rf=Rf, freq=freq, window=window)
    treynor = ( risk_premium / betas ).dropna()
    return treynor


def get_treynor_ratio_df(returns:pd.DataFrame, betas = None, freq=None, window=None, verbose=0):
    pad.verbose('- Treynor-ratio -', level=2, verbose=verbose)
    result = pd.DataFrame(columns = returns.columns, index = returns.index)
    Rf = nefin_risk_free(freq=freq)
    if betas is None or type(betas) is int:
        betas = get_betas(betas, freq, verbose=verbose)
    for i in range(returns.shape[1]):
        portfolio = returns.columns[i]
        pad.verbose(f'{i}. Índice de Treynor --- frequência {freq} --- janela {window} --- Faltam {returns.shape[1]-i}', level=5, verbose=verbose)
        result[portfolio] = get_treynor_ratio(returns[portfolio], betas[portfolio], Rf=Rf, freq=freq, window=window)

    pad.verbose('line', level=2, verbose=verbose)
    return result.dropna(how='all')

def get_betas(window, freq, verbose=0):
    pad.verbose('Buscando betas de mercado', level=4, verbose=verbose)
    freq_dict = {'D' : 'day', 'M' : 'month', 'Q' : 'quarter', 'Y' : 'year' }
    window = f'{window}m' if window is not None else 'all_period'
    path = f'./data/alphas/{freq_dict[freq]}/{window}/'

    betas = pd.DataFrame({file.split('.')[0] : util.df_datetimeindex( pd.read_csv(path+file, index_col=0) )['beta_Market'] for file in util.get_files(path)}, index = pd.date_range(start='2000-01-01', end='2020-12-31', freq=freq) ).dropna(how='all')

    return betas


'''
    VOLATILITY
'''
 
def volatility_df(returns:pd.DataFrame, method="std", window=None, verbose=0):
    pad.verbose(f'- Volatility: {method} -', level=2, verbose=verbose)
    result = pd.DataFrame(columns = returns.columns, index = returns.index)
    for i in range(returns.shape[1]):
        portfolio = returns.columns[i]
        pad.verbose(f'{i}. Volatilidade --- método {method} --- janela {window} --- Faltam {returns.shape[1]-i}', level=5, verbose=verbose)
        result[portfolio] = volatility(returns[portfolio], method=method, window=window)
    
    pad.verbose('line', level=2, verbose=verbose)
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
    returns = returns.dropna()
    result = None
    if len(returns) >= 2:
        avg = st.mean(returns)
        down_values = returns[returns < avg]
        if len(down_values) >= 2:
            result = st.stdev(down_values, xbar=avg)
    return result


'''
    INFORMATION RATIO
'''
def information_ratio_df(returns:pd.DataFrame, freq=None, window=None, verbose=0):
    pad.verbose('- Information-ratio -', level=2, verbose=verbose)
    result = pd.DataFrame(columns = returns.columns, index = returns.index)
    mkt_returns = nefin_single_factor(factor="Market", freq=freq)
    for i in range(returns.shape[1]):
        portfolio = returns.columns[i]
        pad.verbose(f'{i}. Information-Ratio --- janela {window} --- Faltam {returns.shape[1]-i}', level=5, verbose=verbose)
        result[portfolio] = information_ratio(returns[portfolio], mkt_returns, window=window)
    
    pad.verbose('line', level=2, verbose=verbose)
    return result.dropna(how='all')

def information_ratio(returns:pd.Series, mkt_returns=None, window=None):
    difference = (returns - mkt_returns)
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

def get_return_df(returns:pd.DataFrame, window=None, verbose=0):
    pad.verbose('- Retornos Acumulados -', level=2, verbose=verbose)
    result = pd.DataFrame(columns = returns.columns, index = returns.index)
    for i in range(returns.shape[1]):
        portfolio = returns.columns[i]
        pad.verbose(f'{i}. Retorno acumulado --- janela {window} --- Faltam {returns.shape[1]-i}', level=5, verbose=verbose)
        result[portfolio] = get_return(returns[portfolio], window=window)

    pad.verbose('line', level=2, verbose=verbose)
    return result.dropna(how='all')


'''
    Trailing sum
'''

def trailing_sum(serie, window = None):
    serie = serie.dropna()
    if len(serie) > 0:
        if window is None:
            s = [ serie[0 : i].sum() for i in range(1, serie.shape[0]+1)]
            window = 1
        else:
            s = [serie[i-window : i].sum() for i in range(window, serie.shape[0]+1)]
        result = pd.Series(s, index = serie.index[(window-1): serie.shape[0]])
    else:
        result = pd.Series([], index=serie.index)
    return result

def trailing_sum_df(data, window = None, verbose=0):
    pad.verbose('- Trailing sum -', level=2, verbose=verbose)
    result = pd.DataFrame(columns = data.columns, index = data.index)
    for i in range(data.shape[1]):
        portfolio = data.columns[i]
        pad.verbose(f'{i}. Trailing sum --- janela {window} --- Faltam {data.shape[1]-i}', level=5, verbose=verbose)
        s = trailing_sum(data[portfolio], window=window)
        result[portfolio] = s
    pad.verbose('line', level=2, verbose=verbose)
    return result.dropna(how='all')


