import pandas as pd
import datetime as dt
from factors.factors import get_fatores
from funds.preprocess_fias import preprocess_fis
from funds.alpha import jensens_alpha
from funds.characteristics import extract_characteristics
from causality.granger import granger_causality, granger_scores

from util import util, padding as pad


def granger_routine(freqs = ['M', 'Q'], windows = [None, 12, 24, 36], verbose=0, persist=False):    
    freq_dic = {'M':'month', 'Q':'quarter', 'Y':'year', 'D':'day'}
    aprouved_funds = apply_funds_filter(years=5)
    results = dict()
    for freq in freqs:
        for window in windows:
            for separate_lags in [True, False]:
                for binary in [True, False]:
                    characteristics = get_characteristics(freq, window, verbose=verbose)
                    alphas = get_alphas(window, freq, verbose=verbose)

                    gtests = granger_causality(characteristics, alphas, fund_filter = aprouved_funds, maxlag=15, statistical_test='all', separate_lags=separate_lags, binary=binary, verbose=verbose)
                    
                    lags_dir = 'separated_lags' if separate_lags else 'not_separated_lags'
                    binary_dir = 'binary' if binary else 'continuous'
                    if type(gtests) is dict:
                        pad.persist_collection(gtests, path=f'./data/granger_tests/{freq_dic[freq]}/{window}/{lags_dir}/{binary_dir}/', to_persist=persist, _verbose=verbose, verbose_level=2, verbose_str="Persistindo testes de Granger") 
                    else:
                        pad.persist_collection({'granger_results':gtests}, path=f'./data/granger_tests/{freq_dic[freq]}/{window}/{lags}/{binary_dir}/', to_persist=persist, _verbose=verbose, verbose_level=2, verbose_str="Persistindo testes de Granger")

                    #granger scores
                    scores = granger_scores(gtests)
                    pad.persist_collection({f'scores' : scores}, path=f'./data/granger_tests/{freq_dic[freq]}/{window}/{lags}/scores/', to_persist=persist, _verbose=verbose, verbose_level=2, verbose_str="Persistindo scores do testes de Granger")
                    results[(freq, window)] = scores
    return results

def apply_funds_filter(years=5):
    lifetime = util.df_datetimeindex( pd.read_csv('./data/caracteristicas/month/None/lifetime.csv',index_col=0) )
    lifetime = lifetime.iloc[0:lifetime.index.get_loc(dt.datetime(2020,12,31)) + 1]

    life_gt_year = lifetime[lifetime > 365*years].dropna(how='all', axis='columns')
    return set(life_gt_year.columns)


def get_characteristics(freq, window, verbose=0):
    pad.verbose('Buscando características', level=4, verbose=verbose)
    freq_dic = {'M':'month', 'Q':'quarter', 'Y':'year', 'D':'day'}

    path = f'./data/caracteristicas/{freq_dic[freq]}/{window}/'
    files = ['variation','sharpe','treynor','lifetime','information_ratio','standard_deviation','downside_deviation','equity','cotistas','captacao','captacao_liquida','resgate','resgate_IR','resgate_total']

    characteristics = {file.split('.')[0] : util.df_datetimeindex( pd.read_csv(path+file+'.csv', index_col=0) ) for file in files}
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
            if freq == 'Q' and window == 12:
                continue
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
    alphas = {name : util.reindex_timeseries(alphas[name], freq=freq) for name in alphas.keys()}
    
    return alphas

