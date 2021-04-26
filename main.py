import pandas as pd
from routines import routines
from factors.factors import get_fatores
from funds.preprocess_fias import preprocess_fis

from util import util

def main():
    verbose = 5
    windows = [None, 12,24, 36]
    windows = [36]
    fis     = preprocess_fis(pd.read_csv("./data/cotas_fias.csv"), verbose=verbose)
    fatores = get_fatores(source="nefin", verbose=verbose)
    for w in windows:
        routines.granger_routine(fis, fatores, window=w, test = False, persist=True, verbose = verbose)

if __name__ == "__main__":
	main()


