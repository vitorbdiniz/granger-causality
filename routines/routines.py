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
    if test:
        funds_characts = {FI : util.df_datetimeindex(pd.read_csv(f'./data/caracteristicas/{directory}/{FI}.csv', index_col=0)) for FI in fis.keys()}
    else:
        funds_characts = extract_characteristics(alfas, fis, window=window, dropna="any",verbose=verbose)
        pad.persist_collection(funds_characts, path=f'./data/caracteristicas/{directory}/', to_persist=persist, _verbose=verbose, verbose_level=2, verbose_str="Persistindo características") 
    #granger tests
    gtests = granger_tests(funds_characts, alfas, statistical_test='all',verbose=verbose)
    pad.persist_collection(gtests, path=f'./data/granger_tests/{directory}/', to_persist=persist, _verbose=verbose, verbose_level=2, verbose_str="Persistindo testes de Granger") 

    #granger scores
    scores = granger_scores(gtests)
    pad.persist(scores, path=f'./data/granger_tests/{directory}/scores/granger_scores.csv', to_persist=persist, _verbose=verbose, verbose_level=2, verbose_str="Persistindo scores do testes de Granger")


def alpha_routine(fis, fatores, period = 'day', windows = [None, 12,24, 36], verbose=0, persist=False):
    windows_dict = {'day': 22, 'month':1, 'quarter':0.25, 'year':1/12}
    for window in windows:
        rolling_window = f'{int(window)}m' if window is not None else 'all_period'
        if window is not None:
            window = int(window*windows_dict[period])

        #Jensen's alpha
        alfas = jensens_alpha(fatores, fis, janela=window, verbose=verbose)
        pad.persist_collection(alfas, path=f'./data/alphas/{period}/{rolling_window}/', extension=".csv", to_persist=persist, _verbose=verbose, verbose_level=2, verbose_str="Persistindo Alfas")


def characteristics_routine(freqs = ['M', 'Q'], windows = [None, 12, 24, 36], verbose=0, persist=False):
    result = dict()
    for freq in freqs:
        fias_characteristics = get_characts_economatica(freq=freq)
        for window in windows:
            characteristics = extract_characteristics(fias_characteristics, freq = freq, window = window, verbose=0)
            result[(window, freq)] = characteristics
            pad.persist_collection(characteristics, path=f'./data/caracteristicas/{period}/{window}/', extension=".csv", to_persist=persist, _verbose=verbose, verbose_level=2, verbose_str="Persistindo Características")
    return result

def get_characts_economatica(freq='M'):
    pad.verbose('Buscando dados do Economática', level=4, verbose=verbose)
    freq_dic = {'M':'month', 'Q':'quarter', 'Y':'year', 'D':'day'}
    names = ['fis_acc', 'capt', 'capt_liq', 'cotistas', 'PL', 'resgate', 'resgate_IR', 'resgate_total', 'lifetime']
    characteristics = {n : util.df_datetimeindex(pd.read_csv(f'./data/economatica/{freq_dic[freq]}/{n}_economatica_{freq_dic[freq]}.csv', index_col=0)) for n in names}
    return characteristics

def get_alphas(window, freq, verbose=0):
    pad.verbose('Buscando alfas', level=4, verbose=verbose)

    freq_dict = {'D' : 'day', 'M' : 'month', 'Q' : 'quarter', 'Y' : 'year' }
    window = f'{window}m' if window is not None else 'all_period'
    path = f'./data/alphas/{freq_dict[freq]}/{window}/'

    alphas = {file.split('.')[0] : util.df_datetimeindex( pd.read_csv(path+file, index_col=0) ) for file in util.get_files(path)}
    return alphas

