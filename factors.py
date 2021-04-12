import pandas as pd
import numpy as np
import datetime as dt
import util, padding as pad

def get_fatores(source, verbose=0):
    pad.verbose("- Calculando Fatores de Risco -", level=2, verbose=verbose)
    factors = nefin_factors(verbose=verbose) if source == "nefin" else tc_factors()
    pad.verbose("line", level=2, verbose=verbose)
    return factors

def tc_factors():
    return "TODO"

concat_date = lambda date : pd.DatetimeIndex([dt.date(date[0].iloc[i], date[1].iloc[i], date[2].iloc[i]) for i in range(len(date[0]))])

def nefin_factors(verbose=0):
    pad.verbose("Downloading Nefin Factors", level=3, verbose=verbose)
    mkt = pd.read_excel("http://nefin.com.br/Risk%20Factors/Market_Factor.xls") 
    hml = pd.read_excel("http://nefin.com.br/Risk%20Factors/HML_Factor.xls")    
    smb = pd.read_excel("http://nefin.com.br/Risk%20Factors/SMB_Factor.xls")    
    wml = pd.read_excel("http://nefin.com.br/Risk%20Factors/WML_Factor.xls")    
    iml = pd.read_excel("http://nefin.com.br/Risk%20Factors/IML_Factor.xls")    

    risk_factors = pd.DataFrame({"year":mkt["year"], "month":mkt["month"], "day":mkt["day"], "MKT" : mkt["Rm_minus_Rf"], "HML" : hml["HML"], "SMB" : smb["SMB"],"WML" : wml["WML"],"IML" : iml["IML"]})
    risk_factors.index = concat_date([risk_factors["year"], risk_factors["month"], risk_factors['day']])
    risk_factors = risk_factors.drop(columns=["year", "month", "day"])

    return risk_factors

def nefin_single_factor(factor="Market"):
    F = pd.read_excel(f"http://nefin.com.br/Risk%20Factors/{factor}_Factor.xls")
    index = concat_date([ F['year'], F['month'], F['day'] ])
    values = F[ F.columns[-1] ]

    return pd.Series(values, index=index)

def nefin_risk_free():
    nefin_Rf = pd.read_excel('http://nefin.com.br/Risk%20Factors/Risk_Free.xls')
    
    index = concat_date( [ nefin_Rf['year'], nefin_Rf['month'], nefin_Rf['day'] ] )
    values = nefin_Rf['Risk_free'].values
    return pd.Series(values, index=index)


    

