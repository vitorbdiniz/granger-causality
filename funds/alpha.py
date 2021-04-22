import numpy as np
import pandas as pd

import statsmodels.api as sm
from statsmodels.tools import add_constant
from scipy.stats import zscore

from sklearn.model_selection import cross_validate
from sklearn.base import BaseEstimator, RegressorMixin


from funds import preprocess_fias as FI
from util import util, padding as pad



class SMWrapper(BaseEstimator, RegressorMixin):
    """ 
        A universal sklearn-style wrapper for statsmodels regressors
    """
    def __init__(self, model_class, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept
    def fit(self, X, y):
        self.model_ = self.model_class(y, X)
        self.results_ = self.model_.fit()
    def predict(self, X):
        return self.results_.predict(X)
    def get_stats(self):
        return self.results_.params.tolist() + self.results_.pvalues.tolist() + self.results_.tvalues.tolist() + [self.results_.fvalue]+[self.results_.f_pvalue]+[self.results_.rsquared_adj]
    def summary(self):
        return self.results_.summary()




def jensens_alpha(risk_factors, portfolios_returns, janela=None,verbose=0):
    """
        Calcula o Alfa de Jensen para os fundos fornecidos, a partir dos fatores previamente calculados

        Retorna um pandas.DataFrame com colunas: "Nome", "alfa" e os betas solicitados
    """
    pad.verbose("Calculando alfas dos fundos de investimento", level=2, verbose=verbose)
    columns = get_columns(risk_factors.columns)
    alphas = dict()
    i=1
    for each_fund in portfolios_returns:
        pad.verbose(str(i)+". Calculando alfa do Fundo " + str(portfolios_returns[each_fund]["fundo"].iloc[0]) + " ---- faltam "+str(len(portfolios_returns.keys())-i), level=5, verbose=verbose)        
        data = preprocess_dates(portfolios_returns[each_fund], risk_factors)
        if janela:
            alphas[each_fund] = alpha_window_algorithm(data, portfolios_returns[each_fund], columns, window=janela, fund_name=each_fund, verbose_counter=i, verbose=verbose)
        else:
            alphas[each_fund] = alpha_algorithm(data, portfolios_returns[each_fund], columns, fund_name=each_fund, verbose_counter=i, verbose=verbose)
        i+=1
    return alphas

def alpha_algorithm(data, returns, columns, fund_name='', verbose_counter=0, verbose=0):
    df = pd.DataFrame(columns=columns)
    for j in range(20, data.shape[0]):
        pad.verbose(f"{verbose_counter}.{j}. Calculando alfa do Fundo {fund_name} para o dia {data.index[j]} ---- faltam {len(data.index)-j}", level=4, verbose=verbose)
        df.loc[data.index[j]] = get_factor_exposition(data.iloc[0:j+1], fund_name, verbose=verbose)
    return df

def alpha_window_algorithm(data, returns, columns, window=12, fund_name='', verbose_counter=0, verbose=0):
    '''
        Calcula o alfa de um portfolio específico dentro de uma janela mensal móvel
    '''
    window *= 22 #Quantidade média de dias úteis em um por mês em um ano
    df = pd.DataFrame(columns=columns)
    for j in range(window, data.shape[0]):
        pad.verbose(f"{verbose_counter}.{j}. Calculando alfa do Fundo {fund_name} com uma janela móvel de {window/22} meses para o dia {data.index[j]} ---- faltam {len(data.index)-j}", level=4, verbose=verbose)
        df.loc[data.index[j]] = get_factor_exposition(data.iloc[j-window:j+1], fund_name, verbose=verbose)
    return df

def get_factor_exposition(df, name="portfolio", persist=True, verbose=0):
    """
        Realiza a regressão com base nos retornos do portfólio e nos fatores de risco calculados

        retorna uma lista: alfa + betas + tvalores + pvalores + fvalor + pvalor do fvalor + R² ajustado
    """
    data = preprocess_data(df)
    regr = linear_regression(data, target=["cotas"], test_size=0, cv=0, verbose=verbose, persist=persist)
    
    if persist:
        util.write_file(path="./data/alphas/regression_tables/tabela_"+str(name)+": "+str(data.index[-1]) + ".txt", data=str(regr.summary()) )

    return regr.get_stats()

def linear_regression(data, target, test_size = 0.1, cv=0, verbose=0, persist=False):
    y = data[target]
    X = add_constant(data.drop(columns=target))
    model = SMWrapper(sm.OLS)
    if cv != 0 or test_size != 0:
        regr = cross_validate(estimator=model, X=X, y=y, cv=7, n_jobs=-1, return_estimator=True, verbose=0)
        best_model_score, index = -9999999,0
        for i in range(len(regr["test_score"])):
            if regr["test_score"][i] > best_model_score:
                best_model_score = regr["test_score"][i]
                index = i
        best_model = regr["estimator"][index]
    else:
        model.fit(X,y)
        best_model = model
    return best_model




'''
    FUNÇÕES AUXILIARES
'''



def preprocess_data(data):
    data = util.drop_duplicate_index(data)
    data = outlier_treatment(data, method="iqr", quantile=0.25, mult=1.5)
    return data

def outlier_treatment(df, method, quantile=0.25, mult=1.5):
    if method == "iqr":
        result = outlier_treatment_by_iqr(df, quantile=quantile, mult=mult)
    else:
        result = outlier_treatment_by_zscore(df, mult=mult)
    return result

def outlier_treatment_by_zscore(df, mult=1.5):
    outliers = set()
    for factors in df:
        z = zscore(df[factors])
        for i in range(len(z)):
            if abs(z[i]) > mult:
                outliers.add(i)
    result = df.drop(outliers, axis="index")
    return result

def outlier_treatment_by_iqr(data, quantile=0.25, mult=1.5):
    cols = data.columns.tolist()
    outliers = set()
    for fac in cols:
        q75 = np.quantile([ float(x) for x in data[fac]], 1-quantile)
        q25 = np.quantile([ float(x) for x in data[fac]], quantile)
        iqr = q75 - q25
        upper, lower = q75 + iqr*mult , q25 - iqr*mult
        for i in data.index:
            num = data[fac].loc[i].iloc[-1] if type(data[fac].loc[i]) == type(pd.Series([])) else data[fac].loc[i]
            if num > upper or num < lower:
                outliers.add(i)
    result = data.drop(outliers, axis="rows")
    return result

def get_columns(fatores):
    fatores = ["alpha"] + list(fatores)
    statistics = ["alpha"] + [f"beta_{f}" for f in fatores if f != "alpha"]
    statistics += [f"pvalue_{f}" for f in fatores]
    statistics += [f"tvalue_{f}" for f in fatores]
    statistics += ["fvalue", "f_pvalue", "R_squared_adj"]

    return statistics



def preprocess_dates(fundo, fatores):
    '''
        Compara datas entre os dataframes de fundos e fatores, mantendo apenas datas existentes nos dois.
    '''
    fatores.dropna(inplace=True)
    fundo.dropna(inplace=True)
    nome,cotas,MKT, SMB, HML, IML, WML, dates = [],[],[],[],[],[],[],[]
    for i in range(fundo.shape[0]):
        if fundo["data"].iloc[i] in fatores.index:
            cotas += [fundo["variacao"].iloc[i]]
            MKT    += [fatores.loc[fundo["data"].iloc[i]]["Market"]]
            SMB    += [fatores.loc[fundo["data"].iloc[i]]["SMB"]]
            HML      += [fatores.loc[fundo["data"].iloc[i]]["HML"]]
            IML   += [fatores.loc[fundo["data"].iloc[i]]["IML"]]
            WML   += [fatores.loc[fundo["data"].iloc[i]]["WML"]]
            #BAB      += [fatores.loc[fundo["data"].iloc[i]]["BAB"]]
            #QMJ += [fatores.loc[fundo["data"].iloc[i]]["QMJ"]]


            dates += [fundo["data"].iloc[i]]
            nome += [fundo["fundo"].iloc[i]]

    result = pd.DataFrame({"cotas": cotas, "MKT": MKT, "SMB": SMB, "HML": HML, "IML" : IML, "WML" : WML},index=dates)
    return result.drop(labels=result.index[0], axis="index")