import pandas as pd

from factors import get_fatores
from preprocess_fias import preprocess_fis
from alpha import jensens_alpha
from characteristics import extract_characteristics

import util, padding as pad


def main():
    verbose = 5
    persist=True
    test = True

    fis = preprocess_fis(pd.read_csv("./data/cotas_fias.csv"), verbose=verbose)
    fatores = get_fatores(source="nefin", verbose=verbose)
    if test:
        alfas = {FI : util.df_datetimeindex(pd.read_csv(f'./data/alphas/{FI}.csv', index_col=0)) for FI in fis.keys()}
    else:
        alfas = jensens_alpha(fatores, fis, verbose=verbose)

    pad.persist_collection(alfas, path='./data/alphas/', extension=".csv", to_persist=persist, _verbose=verbose, verbose_level=0, verbose_str="Persistindo Alfas")

    #caracteristicas
    funds_characts = extract_characteristics(alfas, fis)
    pad.persist(funds_characts, path='./data/caracteristicas/characteristics.csv', to_persist=persist, _verbose=verbose, verbose_level=2, verbose_str="Persistindo características")
    print(funds_characts)
    #granger



if __name__ == "__main__":
	main()    


