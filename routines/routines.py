import pandas as pd

from factors.factors import get_fatores
from funds.preprocess_fias import preprocess_fis
from funds.alpha import jensens_alpha
from funds.characteristics import extract_characteristics
from causality.granger import granger_tests, granger_scores

from util import util, padding as pad


def granger_routine(characteristics=None, alphas=None, freqs = ['M', 'Q'], windows = [None, 12, 24, 36], verbose=0, persist=False):    
    freq_dic = {'M':'month', 'Q':'quarter', 'Y':'year', 'D':'day'}
    results = dict()
    for freq in freqs:
        for window in windows:
            if characteristics is None:
                characteristics = get_characteristics(freq, window)
            if alphas is None:
                alphas = get_alphas(window, freq, verbose=verbose)
            gtests = granger_tests(characteristics, alphas, statistical_test='all', verbose=verbose)
            pad.persist_collection(gtests, path=f'./data/granger_tests/{freq_dic[freq]}/{window}/', to_persist=persist, _verbose=verbose, verbose_level=2, verbose_str="Persistindo testes de Granger") 

            #granger scores
            scores = granger_scores(gtests)
            pad.persist_collection({f'scores_{freq_dic[freq]}_{window}m' : scores}, path=f'./data/granger_tests/scores/', to_persist=persist, _verbose=verbose, verbose_level=2, verbose_str="Persistindo scores do testes de Granger")
            results[(freq, window)] = scores
    return results


def get_characteristics(freq, window):
    freq_dic = {'M':'month', 'Q':'quarter', 'Y':'year', 'D':'day'}
    path = f'./data/caracteristicas/{freq_dic[freq]}/{window}/'
    characteristics = {file.split('.')[0] : util.df_datetimeindex( pd.read_csv(path+file, index_col=0) ) for file in util.get_files(path)}
    return characteristics

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
    freq_dic = {'M':'month', 'Q':'quarter', 'Y':'year', 'D':'day'}
    result = dict()
    for freq in freqs:
        fias_characteristics = get_characts_economatica(freq=freq, verbose=verbose)
        for window in windows:
            characteristics = extract_characteristics(fias_characteristics, freq = freq, window = window, verbose=verbose)
            result[(window, freq)] = characteristics
            pad.persist_collection(characteristics, path=f'./data/caracteristicas/{freq_dic[freq]}/{window}/', extension=".csv", to_persist=persist, _verbose=verbose, verbose_level=2, verbose_str="Persistindo Características")
    return result

def get_characts_economatica(freq='M', verbose=0):
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

