import pandas as pd
import numpy as np
import statistics as st
import datetime as dt

from factors import nefin_risk_free, nefin_single_factor

import padding as pad

def extract_characteristics(alfas, fis, verbose=0):
    characteristics = { 
        FI: pd.DataFrame({
            'sharpe' : sharpe_index(fis[FI]['variacao'], verbose=verbose),
            'treynor' : treynor_index(fis[FI]['variacao'], alfas[FI]['beta_MKT'], verbose=verbose),
            'lifetime' : get_lifetime(fis[FI]['variacao'], verbose=verbose),
            'information' : information_ratio(fis[FI]['variacao'], verbose=verbose),
            'vol' : volatility(fis[FI]['variacao'], method="std", verbose=verbose),
            'vol_dsd' : volatility(fis[FI]['variacao'], method="dsd", verbose=verbose)
            },
            index = fis[FI].index) 
        for FI in fis.keys() 
    }
    return pd.DataFrame(characteristics, index = characteristics['sharpe'])

def capm_risk_premium(returns, verbose=0):
    pad.verbose("Calculando Prêmio pelo risco de Mercado", level=2, verbose=verbose)
    return (returns - nefin_risk_free()).dropna()

def sharpe_index(returns, verbose=0):
    pad.verbose("Calculando Índice de Sharpe", level=2, verbose=verbose)
    return ( capm_risk_premium(returns) / volatility(returns) ).dropna()

def treynor_index(returns, betas, verbose=0):
    pad.verbose("Calculando Índice de Treynor", level=2, verbose=verbose)
    return ( capm_risk_premium(returns) / betas ).dropna()

def get_lifetime(returns, verbose=0):
    pad.verbose("Calculando Tempo de vida", level=2, verbose=verbose)
    return range(1, (returns.index[ returns.shape[0] ] - returns.index[0]).days +1)  

def leverage():
    return #TODO


def downside_deviation(returns):
    s = 0
    c = 0
    avg = st.mean(returns)
    for r in returns:
        if r < avg:
            c += 1
            s += (r - avg)**2
    dsd = s/c
    return dsd




def volatility(returns, method="std", verbose=0):
    """
        Calculate volatility by the sample's standard deviation(std) or by its downside deviation (dsd)
        method: {'std', 'dsd'}
        returns: float
    """
    pad.verbose(f"Calculando Volatilidade: método {method}", level=2, verbose=verbose)
    if method == "std":
        vol = pd.Series( [(returns.iloc[0:i]).std() for i in range(1, returns.shape[0]+1)], index = returns.index )
    elif method == "dsd":
        vol = pd.Series([downside_deviation(returns.iloc[0:i]) for i in range(1, returns.shape[0]+1)], index = returns.index)

    return vol

def tracking_error(returns, MKT, verbose=0):
    pad.verbose("Calculando Tracking Error", level=2, verbose=verbose)
    return volatility( (returns-MKT).dropna() )
    
def information_ratio(returns, verbose=0):
    pad.verbose("Calculando Information Ratio", level=2, verbose=verbose)
    MKT = nefin_single_factor(factor="Market")
    return ((returns - MKT).dropna() / tracking_error(returns, MKT)).dropna()

