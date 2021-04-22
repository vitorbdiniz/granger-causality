import pandas as pd

from factors import get_fatores
from preprocess_fias import preprocess_fis
from alpha import jensens_alpha
from characteristics import extract_characteristics
from granger import granger_tests, granger_scores

import util, padding as pad


def granger_cumulative(test = False, persist=True, verbose = 5):
    fis = preprocess_fis(pd.read_csv("./data/cotas_fias.csv"), verbose=verbose)
    fatores = get_fatores(source="nefin", verbose=verbose)
    
    if test:
        alfas = {FI : util.df_datetimeindex(pd.read_csv(f'./data/alphas/{FI}.csv', index_col=0)) for FI in fis.keys()}
    else:
        alfas = jensens_alpha(fatores, fis, verbose=verbose)
        pad.persist_collection(alfas, path='./data/alphas/', extension=".csv", to_persist=persist, _verbose=verbose, verbose_level=2, verbose_str="Persistindo Alfas")

    #caracteristicas
    funds_characts = extract_characteristics(alfas, fis, dropna="any",verbose=verbose)
    pad.persist_collection(funds_characts, path='./data/caracteristicas/', to_persist=persist, _verbose=verbose, verbose_level=2, verbose_str="Persistindo características") 
    
    #granger
    gtests = granger_tests(funds_characts, alfas, statistical_test='all',verbose=verbose)
    pad.persist_collection(gtests, path='./data/granger_tests/', to_persist=persist, _verbose=verbose, verbose_level=2, verbose_str="Persistindo testes de Granger") 

    #granger scores
    scores = granger_scores(gtests)
    pad.persist(scores, path='./data/granger_tests/scores/granger_scores.csv', to_persist=persist, _verbose=verbose, verbose_level=2, verbose_str="Persistindo scores do testes de Granger")



def granger_window(window=12,test = False, persist=True, verbose = 5):
    '''
        Realiza testes de Granger para séries temporais de `window` meses
    '''
    fis = preprocess_fis(pd.read_csv("./data/cotas_fias.csv"), verbose=verbose)
    fatores = get_fatores(source="nefin", verbose=verbose)

    if test:
        alfas = {FI : util.df_datetimeindex(pd.read_csv(f'./data/alphas/{FI}.csv', index_col=0)) for FI in fis.keys()}
    else:
        alfas = jensens_alpha(fatores, fis, verbose=verbose)
        pad.persist_collection(alfas, path='./data/alphas/', extension=".csv", to_persist=persist, _verbose=verbose, verbose_level=2, verbose_str="Persistindo Alfas")

    #caracteristicas
    funds_characts = extract_characteristics(alfas, fis, dropna="any",verbose=verbose)
    pad.persist_collection(funds_characts, path='./data/caracteristicas/', to_persist=persist, _verbose=verbose, verbose_level=2, verbose_str="Persistindo características") 
    
    #granger
    gtests = granger_tests(funds_characts, alfas, statistical_test='all',verbose=verbose)
    pad.persist_collection(gtests, path='./data/granger_tests/', to_persist=persist, _verbose=verbose, verbose_level=2, verbose_str="Persistindo testes de Granger") 

    #granger scores
    scores = granger_scores(gtests)
    pad.persist(scores, path='./data/granger_tests/scores/granger_scores.csv', to_persist=persist, _verbose=verbose, verbose_level=2, verbose_str="Persistindo scores do testes de Granger")
