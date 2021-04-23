import pandas as pd
import numpy as np
import datetime as dt

from util import util, padding as pad

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
    
    for FI in fis:
        fis[FI].index = fis[FI]['data']
        fis[FI] = util.drop_duplicate_index(fis[FI])
        fis[FI] = util.df_datetimeindex(fis[FI])
    

    pad.verbose("line", level=2, verbose=verbose)
    return fis
