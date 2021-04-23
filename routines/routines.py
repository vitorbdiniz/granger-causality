import pandas as pd

from factors.factors import get_fatores
from funds.preprocess_fias import preprocess_fis
from funds.alpha import jensens_alpha
from funds.characteristics import extract_characteristics
from causality.granger import granger_tests, granger_scores

from util import util, padding as pad


def granger_routine(fis=None, fatores=None, window = None, test = False, persist=True, verbose = 5):    
    if fis is None:
        fis = preprocess_fis(pd.read_csv("./data/cotas_fias.csv"), verbose=verbose)
    if fatores is None:
        fatores = get_fatores(source="nefin", verbose=verbose)

    directory = f'{int(window)}m' if window is not None else 'all_period'
    
    #Jensen's alpha
    if test:
        alfas = {FI : util.df_datetimeindex(pd.read_csv(f'./data/alphas/{directory}/{FI}.csv', index_col=0)) for FI in fis.keys()}
    else:
        alfas = jensens_alpha(fatores, fis, janela=window, verbose=verbose)
        pad.persist_collection(alfas, path=f'./data/alphas/{directory}/', extension=".csv", to_persist=persist, _verbose=verbose, verbose_level=2, verbose_str="Persistindo Alfas")

    #caracteristicas
    funds_characts = extract_characteristics(alfas, fis, window=window, dropna="any",verbose=verbose)
    pad.persist_collection(funds_characts, path=f'./data/caracteristicas/{directory}/', to_persist=persist, _verbose=verbose, verbose_level=2, verbose_str="Persistindo características") 
    
    #granger tests
    gtests = granger_tests(funds_characts, alfas, statistical_test='all',verbose=verbose)
    pad.persist_collection(gtests, path=f'./data/granger_tests/{directory}/', to_persist=persist, _verbose=verbose, verbose_level=2, verbose_str="Persistindo testes de Granger") 

    #granger scores
    scores = granger_scores(gtests)
    pad.persist(scores, path=f'./data/granger_tests/{directory}/scores/granger_scores.csv', to_persist=persist, _verbose=verbose, verbose_level=2, verbose_str="Persistindo scores do testes de Granger")

