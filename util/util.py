from os.path import realpath
import pandas as pd
import numpy as np
import datetime as dt
import pandas_datareader as web
from os import listdir
from os.path import isfile, join

from util import padding as pad

"""

    DICTIONARIES AND DATAFRAMES

"""

def rearange_prices(prices, start=dt.date(2010,1,1), end=dt.date.today(), column = "Close"):
    return pd.DataFrame({ ticker : prices[ticker][column] for ticker in prices.keys() }, index=date_range(start, end, frequency="D"))

def kill_duplicates(df, check_column="index"):
    if check_column == "index":
        validate = list(df.index)
    else:
        validate = list(df[check_column])
    
    df["index"] = df.index.copy()
    df.index = range(0, len(df.index))

    ( duplicates, checked ) = ( set(), set() )
    for i in df.index:
        if validate[i] not in checked:
            checked.add( validate[i] )
        else:
            duplicates.add(i)
    result = df.drop(labels=duplicates, axis="index")
    result.index = result["index"]
    result = result.drop(columns="index")
    return result


"""

        PANDAS SERIES AND DATAFRAMES 

"""
def reindex_timeseries(df, start=None, end=None, freq='M'):
    if type(df) not in (pd.Series, pd.DataFrame):
        raise AttributeError(f'Tipo de `df` deveria ser {pd.Series} ou {pd.DataFrame} e é {type(df)}')
    if df.shape[0] == 0:
        return df.copy()
    if start is None:
        start = df.index[0]
    if end is None:
        end = df.index[-1]

    new_index = pd.date_range(start, end, freq=freq)
    if type(df) is pd.DataFrame:
        result = pd.DataFrame(df, index=new_index)
    elif type(df) is pd.Series:
        result = pd.Series(df, index=new_index)
        
    return result

def df_datetimeindex(df):
    df.index = index_to_datetimeindex(df.index)
    return df

def index_to_datetimeindex(index):
    return pd.DatetimeIndex( [ dt.datetime(year = get_year(d) , month= get_month(d) , day= get_day(d) ) for d in index ] )

def check_index_duplicate(df):
    return drop_duplicate_index(df).index.tolist() != df.index.tolist()

def drop_duplicate_index(df):
    return df[~df.index.duplicated(keep='first')]


def get_previous_data(series, index, dropna=False):
    if dropna:
        series = series.dropna()
    
    if len(series) == 0:
        return 0

    try:
        i = series.index.get_loc(index, method="pad")
    except:
        i = 0
        
    return series.iloc[i]


"""

    TIME MANIPULATION

"""

def count_quarter_days(start, end):
    '''
        Conta os dias úteis por trimestre em um intervalo [start, end]
    '''
    time = getUtilDays(start, end)
    quarters = getQuarterRange(start, end)
    qdays = {q: 0 for q in quarters}
    for t in time:
        qdays[getQuarter(t)] += 1
    return qdays


def nextQuarter(quarter):
    '''
        Retorna o trimestre que segue o trimestre fornecido

        Ex.: nextQuarter("2T2020") -> "3T2020"
    '''
    q = quarter.split("T")
    nextq = 0
    nexty = 0
    if q[0] == '4':
        nextq = 1
        nexty = int(q[1]) + 1
    else:
        nextq = nexty = int(q[0])+1
        nexty = int(q[1])
    return str(nextq)+"T"+str(nexty)


def count_year_days(dates):
    '''
        Conta dias úteis existentes em uma lista de datas
    '''

    res = []
    days = 0
    year = dates[0][0:4]
    for d in dates:
        if d[0:4] == year:
            days += 1
        else:
            res += [days]
            days = 1
            year = d[0:4]

    return res + [days]

def days_per_year(date_array):
    result = dict()
    for d in date_array:
        result[d.year] = result[d.year] + 1 if d.year in result else 1
    return result


def getQuarter(date):
    '''
        Converte string de data para formato trimestral.

        Ex.: getQuarter("2020-05-22") -> "2T2020"
        Ex.: getQuarter("2001-11-09T00:12:45Z") -> "4T2001"
    '''
    d = str(date).split("-")  # d[0] == year; d[1] == month; d[2] == day
    if int(d[1]) <= 3:
        q = "1"
    elif int(d[1]) <= 6:
        q = "2"
    elif int(d[1]) <= 9:
        q = "3"
    else:
        q = "4"
    return q + "T" + d[0]


def compareQuarters(q1, q2):
    '''
        Compara 2 trimestres.

        Ex.: compareQuarters("2T2020", "1T2020") -> 1
        Ex.: compareQuarters("1T2001", "4T2000") -> -1
    '''

    firstQ = q1.split("T")
    lastQ = q2.split("T")
    return 4*(int(firstQ[1]) - int(lastQ[1])) + (int(firstQ[0]) - int(lastQ[0]))

def compareTime(t1, t2, freq):
    if freq == "quarterly":
        res = compareQuarters(t1, t2)
    elif freq == "annually":
        res = t1-t2
    elif freq == "daily" or "monthly":
        res = (dt.date(int(t1[0:4]), int(t1[5:7]), int(t1[8:10])) - dt.date(int(t2[0:4]), int(t2[5:7]), int(t2[8:10]))).days
    else:
        raise AttributeError("Frequência não estipulada corretamente")
    return res

def getNextPeriod(time, freq):
    if freq == "quarterly":
        res = nextQuarter(time)
    elif freq == "annually":
        res = int(time)+1
    elif freq == "daily":
        res = str(dt.date(int(time[0:4]), int(time[5:7]), int(time[8:10]))+ dt.timedelta(days=1))
    else:
        raise AttributeError("Frequência não estipulada corretamente")
    return res

def date_range(start, end, frequency="D"):
    start = dt.datetime(start.year, start.month, start.day)
    end = dt.datetime(end.year, end.month, end.day)
    if frequency == "D":
       result = pd.DatetimeIndex([start + dt.timedelta(days=i) for i in range( (end-start).days+1 )])
    elif frequency == "Y":
       result = pd.DatetimeIndex([ dt.date(i, 1, 1) for i in range( start.year, end.year+1 )])
    elif frequency == "M":
        result = []
        for y in range(start.year, end.year+1):
            for m in range(1, 13):
                if y == start.year and m < end.month:
                    continue
                elif y == end.year and m > end.month:
                    break
                else:
                    result.append( dt.datetime(year=y, month=m, day=1) )
        result = pd.DatetimeIndex(result)
    else:
        raise AttributeError("frequency")
    return result

def getQuarterRange(start=dt.date.today(), end=dt.date.today()):
    '''
        Retorna trimestres existentes em um dado intervalo de datas
        Ex.: getQuarterRange("2019-11-01", "2020-12-20") -> ["4T2019","1T2020","2T2020","3T2020","4T2020"]
    '''

    firstQ = getQuarter(start)
    lastQ = getQuarter(end)
    res = []
    for y in range(get_year(start), get_year(end)+1):
        for q in range(1, 5):
            quarter = str(q)+"T"+str(y)
            if compareQuarters(quarter, firstQ) >= 0 and compareQuarters(quarter, lastQ) <= 0:
                res.append(quarter)
    return res


def getUtilDays(start, end, form="date"):
    '''
        Busca dias úteis em um intervalo dado
    '''
    selic = getSelic(start, end)
    util = list(selic.index)
    if form == "str":
        util = [str(x) for x in util]
    return util


def dateReformat(date, toUsual=True, form="date"):
    if toUsual:
        d = str(date).split('-')[::-1]
        d = d[0] + "/" + d[1] + "/" + d[2]
    else:
        d = str(date).split("/")[::-1]
        if form == "str":
            d = d[0] + "-" + d[1] + "-" + d[2]
        elif form == "date":
            d = dt.date(int(d[0]), int(d[1]), int(d[2]))
    return d

def datesReformat(dates, toUsual=True, form="date"):
    res = [dateReformat(date, toUsual, form=form) for date in dates]
    return res


def getYears(start, end):
    s = int(start[0:4])
    e = int(end[0:4])
    return [i for i in range(s, e+1)]




def transform(date, freq):
    if freq == "quarterly":
        result = getQuarter(str(date))
    elif freq == "annually":
        result = get_year(str(date))
    else:
        result = date
    return result

def get_month(date):
    '''
        retorna o mês de uma data

        Ex.: get_month("2005-01-05") -> 1
    '''
    return int(date[5:7])

def get_day(date):
    '''
        retorna o dia de uma data

        Ex.: get_day("2005-01-05") -> 5
    '''
    return int(date[8:10])

def get_year(date):
    '''
        retorna o ano de uma data

        Ex.: get_year("2005-01-05") -> 2005
    '''
    return int(date[0:4])    

def str_to_date(string):
    return dt.date(year=get_year(string), month=get_month(string), day=get_day(string))

def list_str_to_list_dates(str_list):
    return [str_to_date(s) for s in list(str_list)]


def count_month_days(dates):
    result = []
    n = 0
    last_month = get_month(dates[0])
    for d in dates:
        if last_month == get_month(d):
            n+=1
        else:
            last_month = get_month(d)
            result.append(n)
            n = 1
    result.append(n)
    return result

def get_frequency(start = dt.date.today(), end = dt.date.today(), freq = "daily"):
    '''
        Retorna uma tupla (index, days_number), em que index representa uma lista temporal na frequência desejada e days_number, a quantidade de dias dentro de cada intervalo de tempo
        freq == "daily" or freq == "monthly" or freq == "quarterly" or freq == "annually"
    '''
    if freq == "daily":
        index = getUtilDays(start, end)
        days_number = [1 for i in index]
    elif freq =="monthly":
        index = getUtilDays(str(start), str(end))
        days_number = count_month_days(index)
        index = [dt.datetime(get_year(x), get_month(x), get_day(x)) for x in index]
        index = [ x.date().__str__() for x in pd.DataFrame(index, index=index).resample("M").pad().index]
    elif freq == "quarterly":
        index = getQuarterRange(str(start), str(end))
        days_number = [d for d in count_quarter_days(str(start), str(end)).values()]
    elif freq == "annually":
        index = getYears(str(start), str(end))
        days_number = count_year_days(getUtilDays(str(start), str(end)))
    else:
        raise AttributeError("Frequência não estipulada corretamente")
    return index, days_number


"""

    RETURNS

"""

def getReturns(prices, form="DataFrame"):
    prices = prices.dropna()
    if len(prices) > 0:
        if type(prices) == type(pd.DataFrame({})):
            r = [None]+[prices["Close"].iloc[i] / prices["Close"].iloc[i-1] -1 for i in range(1, len(prices.index))]
        elif type(prices) == type(pd.Series({})):
            r = [None]+[prices.iloc[i] / prices.iloc[i-1] -1 for i in range(1, len(prices.index))]
    else:
        r = []
    if form == "DataFrame" or form == pd.DataFrame:
        returns = pd.DataFrame({"returns":r}, index=prices.index)
    elif form == "Series" or form == pd.Series:
        returns = pd.Series(r, index=prices.index)
    else:
        returns = r
    return returns
    
def allReturns(prices = dict()):
    return pd.DataFrame({ticker:pd.DataFrame(getReturns(prices[ticker]), index=list(prices[ticker].index)) for ticker in prices.keys()}, index=date_range(dt.datetime(2010,1,1), dt.datetime.today()))


def cumulative_return(retornos, return_type = pd.Series):
    '''
        Retorna uma lista ou pandas.Series de retornos acumulados
        retornos: {list, pandas.Series}
        return_type: {pandas.Series, list}
    '''
    capital = 1
    acumulado = []
    for r in retornos:
        if pd.isna(r):
            acumulado += [None]
        else:
            capital = capital*(1+r)
            acumulado += [capital-1]

    if return_type == pd.Series:
        if type(retornos) == pd.Series:
            acumulado = pd.Series(acumulado, index=retornos.index)
        else:
            acumulado = pd.Series(acumulado, index=range(len(retornos)))
    else:
        if len(acumulado) == 0:
            acumulado = [None]
    return acumulado




def avg_return(retornos):

    acc_return = cumulative_return(retornos=retornos)[-1]
    periods = len(retornos) if len(retornos) % 2 == 1 or acc_return > 0 else len(retornos)-1
    avg = (1+acc_return)**(1/periods)-1
    return avg

def mean_annual_return(array):
    cosmos = cumulative_return(array)
    daily_return_cosmos = (cosmos[-1]+1)**(1/len(cosmos))-1
    annual_return_cosmos = (daily_return_cosmos+1)**(250)-1
    return annual_return_cosmos    


def retornos_acumulados_por_periodo(retornos:pd.Series, to_freq='M', calculate_current_freq_returns=False):
    pad.verbose(f'{retornos.name}', level=3, verbose=4)
    retornos = retornos.dropna()
    if calculate_current_freq_returns:
        retornos = returns(retornos)
    if len(retornos) == 0:
        return retornos
    new_index = retornos.resample(to_freq[0].upper()).pad().index
    values = []

    i0 = 0
    for d in new_index:          
        i1 = retornos.index.get_loc(d, method='pad')+1
        acc_return = cumulative_return(retornos[i0:i1]).tolist()
        if len(acc_return) == 0:
            acc_return = [0]
        values.append( acc_return[-1] )
        i0 = i1
    return pd.Series(values, index=new_index)

def retornos_acumulados_por_periodo_df(retornos:pd.DataFrame, to_freq='M', calculate_current_freq_returns=False):
    return pd.DataFrame({col : retornos_acumulados_por_periodo(retornos[col], to_freq, calculate_current_freq_returns) for col in retornos})



"""

    OUTROS CÁLCULOS

"""
def list_to_string(seq, sep=','):
    return sep.join(str(i) for i in seq)

def is_none(val):
    if type(val) == pd.Series or type(val) == pd.DataFrame:
        return not val.any()
    return val == None



def is_iterable(elem):
    try:
        iter(elem)
        return True
    except:
        return False

def getSelic(start = dt.date.today(), end = dt.date.today(), verbose = 0, persist = True, form="DataFrame"):
    pad.verbose("Buscando série histórica da Selic", level=5, verbose=verbose)
    start = dateReformat(str(start))
    end = dateReformat(str(end))

    url = "http://api.bcb.gov.br/dados/serie/bcdata.sgs.11/dados?formato=csv&dataInicial="+ str(start) +"&dataFinal="+str(end)

    start = dateReformat(str(start), toUsual=False, form="date")
    end = dateReformat(str(end), toUsual=False, form="date")

    try:
        selic = pd.read_csv(url, sep=";")
    except:
        selic = pd.DataFrame()

    if "valor" in selic.columns:
        selic["valor"] = [ x/100 for x in reformatDecimalPoint(selic["valor"], to=".")]
        selic.index = datesReformat(selic["data"], False)
        selic = pd.DataFrame({"valor":list(selic["valor"])}, index = selic.index)
    else:
        selic = pd.read_csv("./data/selic.csv", index_col=0)
        selic.index = [dt.date(year=int(d[0:4]), month=int(d[5:7]), day=int(d[8:10])) for d in selic.index]
        selic = selic.loc[start:end]

    selic.index = pd.DatetimeIndex(selic.index)
    if persist:
        selic.to_csv("./data/selic.csv")
    if form == "Series":
        selic = selic["valor"]

    pad.verbose("line", level=5, verbose=verbose)
    return selic




def total_liquidity_per_year(volume, date_array = None, form = "dic", year_form="int"):
    if date_array == None:
        date_array = days_per_year(volume.index)
    result = dict()
    for d in volume.index:
        if pd.isna(volume.loc[d]):
            volume.loc[d] = 0
        result[d.year] = result[d.year] + float(volume.loc[d]) if d.year in result else float(volume.loc[d])
    
    if year_form == "datetime":
        result = { dt.datetime(year=x, month=1, day=1) : result[x] for x in result.keys() }

    if form == "Series":
        result = pd.Series( [result[x] for x in result.keys()] , index = result.keys())
        

    return result


def mean_liquidity_per_year(volume, date_array = None):
    result = total_liquidity_per_year(volume, date_array)
    result = {year : float(result[year])/int(date_array[year])   if year in result else 0   for year in date_array}
    return result

def getCode(ticker):
    ind = -1
    for i in range(len(ticker)):
        if ord(ticker[i]) >= ord('0') and ord(ticker[i]) <= ord('9'):
            ind = i
            break
    return ticker[0:ind]


def findSimilar(ticker, ticker_list):
    code = getCode(ticker).upper()
    similar = []
    for t in ticker_list:
        if code == getCode(t).upper():
            similar += [t]
    return similar


def reformatDecimalPoint(commaNumberList, to="."):
    return [float(commaNumber.replace(",", to)) for commaNumber in commaNumberList]




def moving_average(array, period):
    '''
        Calcula a média móvel para um período selecionado.
    '''
    array = pd.Series(array) if type(array) == type([]) else array

    if period <= 1:
        return array
    MA = array.rolling(window=period).mean()
    NaN = MA[MA.isna()]
    MA = MA[MA.notna()]
    sum_acc = 0
    replaced_NaN = []
    for i in NaN.index:
        sum_acc += array.iloc[i]
        replaced_NaN += [sum_acc/(i+1)]
    NaN = pd.Series(replaced_NaN, index=NaN.index)
    return NaN.append(MA)



def trailing_sum(array, period = 12):
    return [ sum(array[0:i]) for i in range(period)] + [sum(array[i-period:i]) for i in range(period, len(array))]

def get_data_in_year(df, year):
    start = 0
    end = len(df.index)
    year_found = False
    for i in range(len(df.index)):
        if not year_found and df.index[i].year == year:
            start = i
            year_found = True
        elif year_found and df.index[i].year > year:
            end = i
            break
    return df.iloc[start:end].copy()

def none_to_zero(array):
    return [x if pd.notna(x) else 0 for x in array]

    
"""

    STATISTICS

"""

def outlier_treatment_df(data, quantile=0.25, mult=1.5):
    outliers = set()
    for fac in data.columns.tolist():
        outliers = outliers.union( outlier_treatment(data[fac], quantile, mult, out_index=True) )
    result = data.drop(outliers, axis="rows")
    return result


def outlier_treatment(serie, quantile=0.25, mult=1.5, out_index=False):
    '''
        out_index: Boolean -> se True, retona os índices com outliers 
    '''
    outliers = set()

    q75 = np.quantile([ float(x) for x in serie.values], 1-quantile)
    q25 = np.quantile([ float(x) for x in serie.values], quantile)
    iqr = q75 - q25
    upper, lower = q75 + iqr*mult , q25 - iqr*mult
    for i in serie.index:
        num = serie.loc[i].iloc[-1] if type(serie.loc[i]) == pd.Series else serie.loc[i]
        if num > upper or num < lower:
            outliers.add(i)
    if out_index:
        result = outliers
    else:
        result = serie.drop(outliers, axis="rows")
    return result

def merge_lists(lists):
    L = list( set([e for list1 in lists for e in list1]) )
    L.sort()
    return L

def join_series(series_list, dropna = True):
    result = pd.DataFrame({i : series_list[i] for i in range(len(series_list))}, merge_lists( [s.index for s in series_list] )  )
    if dropna:
        result.dropna(inplace=True)
    return result

def rescale_df(df):
    return pd.DataFrame({ col : rescale(df[col]) for col in df.columns})

def rescale(serie):
    if type(serie) == pd.Series:
        index = serie.index
        data = serie.values
    else:
        data = np.array(serie)
        index = range(len(serie))
    minimum = data.min(axis=0)
    maximum = data.max(axis=0)
    result = (data - minimum ) / (maximum - minimum)
    return pd.Series(result, index=index)


def preprocess_serie(serie):
    result = rescale(serie)
    result = outlier_treatment(result)
    return result

def preprocess_series(series_list):
    result = [preprocess_serie(serie) for serie in series_list]
    return join_series(result)


def get_files(directory_path = './'):
    return [f for f in listdir(directory_path) if isfile(join(directory_path, f))]