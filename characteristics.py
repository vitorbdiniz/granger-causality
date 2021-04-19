import pandas as pd
import numpy as np
import statistics as st
import datetime as dt

from factors import nefin_risk_free, nefin_single_factor

import util,padding as pad

def extract_characteristics(alfas, fis, janela = 'all_period', dropna="all", verbose=0):
    """
        
        janela: {'all', 'all_period', '1month', '12months', '24months', '36months'}
    """
    characteristics = { 
        FI: pd.DataFrame({
            'sharpe'             : sharpe_index(fis[FI]['variacao'].dropna(), portfolio=FI, verbose=verbose),
            'treynor'            : treynor_index(fis[FI]['variacao'].dropna(),alfas[FI]['beta_MKT'].dropna(), portfolio=FI, verbose=verbose),
            'lifetime'           : get_lifetime(fis[FI]['variacao'].dropna() , portfolio=FI,verbose=verbose),
            'information-ratio'  : information_ratio(fis[FI]['variacao'].dropna(), portfolio=FI,verbose=verbose),
            'standard_deviation' : volatility(fis[FI]['variacao'].dropna(), portfolio=FI,method="std", verbose=verbose),
            'downside_deviation' : volatility(fis[FI]['variacao'].dropna(), portfolio=FI,method="dsd", verbose=verbose)
            }, index = util.date_range(dt.date(2000,1,1), dt.date.today()) ).dropna(how=dropna)
        for FI in fis.keys() 
    }
    return characteristics

def capm_risk_premium(returns,cumulative = False, verbose=0):
    pad.verbose("Calculando Prêmio pelo risco de Mercado", level=2, verbose=verbose)

    mkt = (returns - nefin_risk_free()).dropna()
    if cumulative:
        mkt = util.cumulative_return(mkt, return_type = pd.Series)
    return mkt

def sharpe_index(returns, portfolio=None,verbose=0):
    pad.verbose(f"Calculando Índice de Sharpe de {portfolio}", level=2, verbose=verbose)
    return ( capm_risk_premium(returns, cumulative = True, verbose=verbose) / volatility(returns) ).dropna()

def treynor_index(returns, betas, portfolio=None,verbose=0):
    pad.verbose(f"Calculando Índice de Treynor de {portfolio}", level=2, verbose=verbose)
    return ( capm_risk_premium(returns, cumulative = True, verbose=verbose) / betas ).dropna()

def get_lifetime(returns, portfolio=None, verbose=0):
    pad.verbose(f"Calculando Tempo de vida de {portfolio}", level=2, verbose=verbose)
    return pd.Series( range(1, returns.shape[0]+1) , index = returns.index)

def leverage():
    return #TODO


def downside_deviation(returns):
    avg = st.mean(returns.values)
    down_values = returns[returns < avg]
    if len(down_values) < 2:
        return None
    return st.stdev(down_values, xbar=avg)


def volatility(returns, method="std", portfolio=None, verbose=0):
    """
        Calculate volatility by the sample's standard deviation(std) or by its downside deviation (dsd)
        method: {'std', 'dsd'}
        returns: float
    """
    pad.verbose(f"Calculando Volatilidade de {portfolio}: método {method}", level=2, verbose=verbose)
    if len(returns) < 2:
        return None

    if method == "std":
        vol = pd.Series( [(returns.iloc[0:i]).std() for i in range(1, returns.shape[0]+1)], index = returns.index )
    elif method == "dsd":
        vol = pd.Series([downside_deviation(returns.iloc[0:i]) for i in range(1, returns.shape[0]+1)], index = returns.index)

    return vol


def information_ratio(returns, portfolio=None, verbose=0):
    pad.verbose(f"Calculando Information Ratio de de {portfolio}", level=2, verbose=verbose)

    MKT = nefin_single_factor(factor="Market")

    premium = util.cumulative_return((returns - MKT).dropna(), return_type=pd.Series)
    fund_tracking_error = volatility( (returns-MKT).dropna() )

    IR = premium / fund_tracking_error

    return IR

