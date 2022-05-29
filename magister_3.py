import urllib.request
import os
from zipfile import ZipFile
import fnmatch
import re
import pandas as pd
import numpy as np
import eurostat
from covid19dh import covid19
import datetime
from time import gmtime, strftime
from dateutil.relativedelta import relativedelta
import covid19poland as PL

currentDateTime = datetime.datetime.now()
current_date = currentDateTime.date()
current_year = int(current_date.strftime("%Y"))


def prepare_eurostat_data(df, feature_name):
    df = df.transpose()
    df = df.iloc[::-1]
    df.index.name = 'Date'
    df['Time'] = df.index
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.rename(columns={0: feature_name})
    df.insert(loc=0, column='Month', value=df['Time'].dt.month)
    df.insert(loc=0, column='Year', value=df['Time'].dt.year)
    df = df.reset_index()
    df = df[['Year', 'Month', feature_name]]
    df[feature_name] = df[feature_name].astype(float)
    return df


def get_data_unemployment(year_from, year_to=current_year):
    filter_pars = {'FREQ': ['M'], 'GEO': ['PL'], 'UNIT': ['THS_PER'], 'INDIC': ['LM-UN-T-TOT'], 'S_ADJ': ['SA']}
    df = eurostat.get_sdmx_data_df('ei_lmhu_m', year_from, year_to, filter_pars=filter_pars, flags=False, verbose=True)
    df.drop(['UNIT', 'S_ADJ', 'INDIC', 'GEO', 'FREQ'], axis=1, errors='ignore', inplace=True)
    df = prepare_eurostat_data(df, 'Bezrobotni')
    df['Bezrobotni'] = df['Bezrobotni']
    return df


def get_data_air(year_from, year_to=current_year):
    filter_pars = {'FREQ': ['M'], 'GEO': ['PL'], 'UNIT': ['FLIGHT'], 'TRA_COV': ['TOTAL'], 'SCHEDULE': ['TOT'],
                   'TRA_MEAS': ['CAF_PAS']}
    df = eurostat.get_sdmx_data_df('avia_paoc', year_from, year_to, filter_pars=filter_pars, flags=False, verbose=True)
    df.drop(['UNIT', 'S_ADJ', 'TRA_COV', 'GEO', 'FREQ', 'SCHEDULE', 'TRA_MEAS'], axis=1, errors='ignore', inplace=True)
    print(df)
    df = prepare_eurostat_data(df, 'Transport lotniczy')
    return df


def get_data_hcip(year_from, year_to=current_year):
    filter_pars = {'FREQ': ['M'], 'GEO': ['PL'], 'UNIT': ['HICP2015'], 'S_ADJ': ['NSA'], 'INDIC': ['CP-HI00']}
    df = eurostat.get_sdmx_data_df('ei_cphi_m', year_from, year_to, filter_pars=filter_pars, flags=False, verbose=True)
    df.drop(['UNIT', 'S_ADJ', 'INDIC', 'GEO', 'FREQ'], axis=1, errors='ignore', inplace=True)
    df = prepare_eurostat_data(df, 'HCIP')
    return df


def get_data_acc(year_from, year_to=current_year):
    #hotels
    # 'I551-I553': 'Hotels; holiday and other short-stay accommodation; camping grounds, recreational vehicle parks and trailer parks'
    filter_pars = {'FREQ': ['M'], 'GEO': ['PL'], 'C_RESID': ['TOTAL'], 'NACE_R2': ['I551'], 'UNIT': ['NR']}
    df = eurostat.get_sdmx_data_df('tour_occ_nim', year_from, year_to, filter_pars=filter_pars, flags=False,
                                   verbose=True)
    df.drop(['UNIT', 'NACE_R2', 'C_RESID', 'GEO', 'FREQ'], axis=1, errors='ignore', inplace=True)
    df = prepare_eurostat_data(df, 'Zakwaterowania')
    return df


def conv_data_deaths_all(file_name, year):
    my_sheet = 'OGÓŁEM'
    df_deaths = pd.read_excel(file_name, sheet_name=my_sheet, header=None)

    my_sheet = 'TYGODNIE ISO8601'
    df_deaths_dates = pd.read_excel(file_name, sheet_name=my_sheet)

    if year == current_year:  # current year is not completed
        df_deaths = df_deaths.iloc[[8]]
        df_deaths = df_deaths.fillna(0)
    else:
        df_deaths = df_deaths.dropna()

    df_deaths = df_deaths.transpose()

    if year < 2020:  # since 2020 file has another look
        df_deaths = df_deaths.rename(columns={df_deaths.columns[1]: 'Liczba zgonów ogółem'})
    else:
        df_deaths = df_deaths.rename(columns={df_deaths.columns[0]: 'Liczba zgonów ogółem'})
    df_deaths = df_deaths.drop(df_deaths.columns.difference(['Liczba zgonów ogółem']), 1)
    df_deaths = df_deaths.drop([0, 1, 2])
    df_deaths = df_deaths.fillna(0.0)
    df_deaths = df_deaths[df_deaths['Liczba zgonów ogółem'] != 0]
    df_deaths.insert(loc=0, column='Week', value=(np.arange(len(df_deaths)) + 1))
    df_deaths['Week'] = df_deaths['Week'].astype(str)
    df_deaths['Week'] = df_deaths['Week'].str.zfill(2)
    df_deaths.insert(loc=0, column='Year', value=str(year))
    df_deaths.insert(loc=1, column='Month', value='')

    if year < 2021:
        df_deaths['Month'] = df_deaths['Year'].astype(str) + '-T' + df_deaths['Week'].astype(str)
    else:
        df_deaths['Month'] = df_deaths['Year'].astype(str) + '-W' + df_deaths['Week'].astype(str)

    df_deaths = df_deaths.reset_index(drop=True)

    for index in range(0, len(df_deaths)):
        week = df_deaths.iloc[index]['Month']
        new_value = df_deaths_dates[df_deaths_dates['TYDZIEŃ'] == week]
        df_deaths.at[index, 'Month'] = new_value.iloc[0]['DATA']

    df_deaths['Month'] = pd.to_datetime(df_deaths['Month'], format='%Y-%m-%d %H:%M:%S', errors='coerce').dt.month
    df_deaths = df_deaths.dropna()
    df_deaths['Month'] = df_deaths['Month'].astype(int)
    df_deaths['Year'] = df_deaths['Year'].astype(int)
    df_deaths['Week'] = df_deaths['Week'].astype(int)
    return df_deaths


def get_data_deaths_all(year_from, year_to=current_year):
    plik = "zgony_wg_tygodni.zip"
    URL = "https://stat.gov.pl/download/gfx/portalinformacyjny/pl/defaultaktualnosci/5468/39/2/1/" + plik
    if not os.path.isfile(plik):
        print('Pobieram plik z ', URL)
        print()
        urllib.request.urlretrieve(URL, plik)
        print('Pobrano plik')
        print()
    else:
        print(f'Plik {plik} już jest na dysku')
        print()

    years = range(year_from, year_to + 1)
    df_deaths_all = pd.DataFrame()

    zip = ZipFile(plik)
    for year in years:
        for info in zip.infolist():
            if re.match(r'.*{}.*\.xlsx$'.format(year), info.filename):
                zip.extract(info)
                df_new = conv_data_deaths_all(file_name=info.filename, year=year)
                df_deaths_all = pd.concat([df_deaths_all, df_new])

    df_deaths_all['Year'] = df_deaths_all['Year'].astype(int)
    return df_deaths_all


def get_data_covid(date_today=current_date):
    df2 = PL.covid_tests()
    df, src = covid19('PL', verbose=False, end=date_today)
    df = df[['date', 'deaths', 'people_fully_vaccinated', 'confirmed']]
    df = df.merge(df2, how='left', on='date')
    df.insert(loc=0, column='Week', value=df['date'].dt.week)
    df.insert(loc=0, column='Month', value=df['date'].dt.month)
    df.insert(loc=0, column='Year', value=df['date'].dt.year)
    df.insert(loc=0, column='Liczba zgonów COVID', value=(df['deaths'] - df['deaths'].shift()))
    df.insert(loc=0, column='Nowe zachorowania', value=(df['confirmed'] - df['confirmed'].shift()))
    df.insert(loc=0, column='Nowe testy', value=(df['tests_all'] - df['tests_all'].shift()))
    df.insert(loc=0, column='Nowi zaszczepieni',
              value=(df['people_fully_vaccinated'] - df['people_fully_vaccinated'].shift()))
    df = df[['Year', 'Month', 'Week', 'Liczba zgonów COVID', 'Nowe zachorowania', 'Nowi zaszczepieni', 'Nowe testy']]
    df = df.fillna(0)
    return df
