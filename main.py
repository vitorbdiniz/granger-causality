import pandas as pd
import datetime as dt

from routines import routines
from factors.factors import get_fatores
from funds.preprocess_fias import preprocess_fis

from util import util, padding as pad

#import warnings
#warnings.filterwarnings("ignore")

def main(routine='alpha', period = 'month'):
    verbose = 5
    persist = True

    if period == 'day' or period == 'month':
        windows = [12, 24, 36, 48, 60, None]
    elif period == 'quarter':
        windows = [24, 36, 48, 60, None]
    elif period == 'year':
        windows = [48,60,None]

    if routine=='alpha':
        fis, fatores = prepare_alpha_routine(period, verbose)
        routines.alpha_routine(fis, fatores, period=period, windows=windows, verbose=verbose, persist=persist)
    
    elif routine == 'characteristics':
        routines.characteristics_routine(freqs = ['M','Q'], windows = [24, 36, 48, 60, None], verbose=verbose, persist=persist)

    else:
        fis = preprocess_fis(pd.read_csv("./data/cotas_fias.csv"), verbose=verbose)
        for w in windows:
            routines.granger_routine(fis, fatores, window=w, test = True, persist=persist, verbose = verbose)

def prepare_alpha_routine(period, verbose):
    if period == 'day':
        fatores = get_fatores(source="nefin", verbose=verbose)
        fis = util.df_datetimeindex(pd.read_csv("./data/retorno_cotas_economatica.csv", index_col=0))
    else:
        fatores = util.df_datetimeindex(pd.read_csv(f'./data/fatores_acc_{period}.csv', index_col=0) )
        fis = util.df_datetimeindex(pd.read_csv(f'./data/fis_acc_{period}.csv', index_col=0) )
    #fatores = util.retornos_acumulados_por_periodo_df(fatores, to_freq=period[0].upper(), calculate_current_freq_returns=False)
    #fatores.to_csv(f'./data/fatores_acc_{period.lower()}.csv')
    return fis, fatores


if __name__ == "__main__":
    #for period in ['quarter', 'year']:
        main(routine='characteristics', period='quarter')


