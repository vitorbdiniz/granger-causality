import pandas as pd
import numpy as np
import statistics as st
import datetime as dt

from factors.factors import nefin_risk_free, nefin_single_factor

from util import util,padding as pad

def extract_characteristics(alfas, fis, window=None, dropna="all", verbose=0):
    '''
        Calcula caracteristicas do dicionário de portfolios `fis` passado dentro de uma janela móvel `window`de tempo.
    '''
    characteristics = dict()
    for FI in fis.keys():
        pad.verbose(f"{FI}. Calculando características", level=5, verbose=verbose)
        characteristics[FI] = pd.DataFrame({
            'sharpe'             : sharpe_index(fis[FI]['variacao'].dropna(), window=window, portfolio=FI, verbose=verbose),
            'treynor'            : treynor_index(fis[FI]['variacao'].dropna(),alfas[FI]['beta_Market'].dropna(), window=window, portfolio=FI, verbose=verbose),
            'lifetime'           : get_lifetime(fis[FI]['variacao'].dropna(), portfolio=FI,verbose=verbose),
            'information-ratio'  : information_ratio(fis[FI]['variacao'].dropna(), window=window, portfolio=FI,verbose=verbose),
            'standard_deviation' : volatility(fis[FI]['variacao'].dropna(), window=window, portfolio=FI,method="std", verbose=verbose),
            'downside_deviation' : volatility(fis[FI]['variacao'].dropna(), window=window, portfolio=FI,method="dsd", verbose=verbose),
            'variacao'           : get_return(fis[FI]['variacao'], window=window, return_type = pd.Series),
            'equity'             : fis[FI]['patrimonio_liquido'],
            'captacao_liquida'   : captacao_liquida_acumulada(fis[FI]['captacao'] - fis[FI]['resgate'], window=window, portfolio=FI, verbose=verbose),
            'cotistas'           : fis[FI]['cotistas']
            }, index = util.date_range(dt.date(2000,1,1), dt.date.today()) ).dropna(how=dropna)
    return characteristics

def captacao_liquida_acumulada(captacao_liquida, window=None, portfolio='', verbose=0):
    s = 0
    acc = []
    pad.verbose(f"- {portfolio}. Captação Líquida", level=3, verbose=verbose)
    for cap in captacao_liquida.values:
        s += cap if pd.notna(cap) else 0
        acc += [s]
    return pd.Series(acc, captacao_liquida.index)


def capm_risk_premium(returns,cumulative = False, window=None, verbose=0):
    pad.verbose("- Prêmio pelo risco de Mercado", level=2, verbose=verbose)
    mkt = (returns - nefin_risk_free()).dropna()
    if cumulative:
        mkt = get_return(mkt, window=window, return_type = pd.Series)
    return mkt

def sharpe_index(returns, window=None, portfolio=None,verbose=0):
    pad.verbose(f"- {portfolio}. Índice de Sharpe", level=2, verbose=verbose)
    risk_premium = capm_risk_premium(returns, cumulative = True, window=window, verbose=0)
    vol = volatility(returns,window=window)
    sharpe = ( risk_premium / vol ).dropna()
    return sharpe

def treynor_index(returns, betas, window=None, portfolio=None,verbose=0):
    pad.verbose(f"- {portfolio}. Índice de Treynor", level=2, verbose=verbose)
    risk_premium = capm_risk_premium(returns, cumulative = True,window=window, verbose=0)
    treynor = (risk_premium / betas).dropna()
    return treynor

def get_lifetime(returns, portfolio=None, verbose=0):
    pad.verbose(f"{portfolio}. Calculando Tempo de vida", level=2, verbose=verbose)
    return pd.Series( range(1, returns.shape[0]+1) , index = returns.index)

def volatility(returns, method="std", window=None, portfolio=None, verbose=0):
    """
        Calculate volatility by the sample's standard deviation(std) or by its downside deviation (dsd)
        method: {'std', 'dsd'}
        cumulative: se cumulative < 0, então é calculada a volatilidade acumulada, senão, é calculada dentro da janela de `cumulative` meses
        returns: pd.Series(dtype=np.float)
    """
    pad.verbose(f"- {portfolio}. Volatilidade: método {method}", level=2, verbose=verbose)
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


def information_ratio(returns, window=None, portfolio=None, verbose=0):
    pad.verbose(f"- {portfolio}. Information Ratio", level=2, verbose=verbose)

    MKT = nefin_single_factor(factor="Market")

    premium = get_return((returns - MKT).dropna(), window=window, return_type=pd.Series)
    fund_tracking_error = volatility( (returns-MKT).dropna(), window=window )

    IR = premium / fund_tracking_error

    return IR

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