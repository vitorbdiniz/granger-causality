import pandas as pd
import numpy as np
import datetime as dt

import util, padding as pad

def preprocess_fis(fundos=pd.DataFrame(), freq="daily", verbose = 0):
    '''
        Recebe um DataFrame com retornos de fundos e retorna um dicionário de fundos
        chaves = codigos
        valores = DataFrames
    '''
    pad.verbose("Pré-processando dados de fundos de investimento", level=2, verbose=verbose)

    fis = dict()
    j=1
    for i in fundos.index:
        pad.verbose(str(j) + ". Pré-processando dados de " +str(fundos["fundo"].loc[i]), level=5, verbose=verbose)
        j+=1
        row = fundos.loc[i]
        if fundos["fundo"].loc[i] not in fis:
            j=1
            fis[fundos["fundo"].loc[i]] = pd.DataFrame(columns=fundos.columns)
        fis[fundos["fundo"].loc[i]].loc[i] = row
    
    fis = process_frequency(fis, freq, verbose)

    pad.verbose("line", level=2, verbose=verbose)
    return fis

def process_frequency(fis=dict(), freq="daily", verbose=False):
    result = dict()
    for fundo in fis:
        fis[fundo].index = pd.DatetimeIndex([dt.date(util.get_year(x), util.get_month(x), util.get_day(x)) for x in fis[fundo]["data"] ])     
        result[fundo] = fis[fundo].resample(freq[0].upper()).pad()
    return result
