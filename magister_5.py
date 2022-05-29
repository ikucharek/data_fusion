import pandas as pd
import numpy as np
from magister_3 import *
from magister_4 import *


# Download medical data
def get_medical_data(year_from):
    df_covid = get_data_covid()  # COVID from 2020
    df_deaths_all = get_data_deaths_all(year_from)  # Deaths all from 2018
    df_return = df_deaths_all.merge(df_covid, how='left', on=['Year', 'Month', 'Week'])
    df_return = df_return.fillna(0.0)
    pd.set_option('display.max_rows', df_return.shape[0] + 1)
    pd.set_option('display.max_columns', df_return.shape[1] + 1)
    # Non COVID deaths calculation
    df_return.insert(loc=4,
                     column='Liczba zgonów nie COVID',
                     value=(df_return['Liczba zgonów ogółem'] - df_return['Liczba zgonów COVID']))
    df_return = df_return.groupby(['Year', 'Month'])[
        'Liczba zgonów nie COVID', 'Liczba zgonów COVID', 'Nowe zachorowania', 'Nowi zaszczepieni', 'Nowe testy'].sum().reset_index()
    df_return = df_return.fillna(0.0)
    return df_return


# Download medical data by week
def get_medical_data_by_week(df_return):
    df_return.groupby(['Year', 'Week'])[
        'Liczba zgonów COVID', 'Nowe zachorowania', 'Nowi zaszczepieni', 'Nowe testy'].sum().reset_index()
    return df_return



# Non COVID deaths by month
def get_ncd_by_month(df_medical):
    df_return = df_medical[['Year', 'Month', 'Liczba zgonów nie COVID']]
    df_return = df_return.groupby(['Year', 'Month'])['Liczba zgonów nie COVID'].sum().reset_index()
    return df_return


def get_economy_data(year_from):
    # HCIP - wskaznik cen konsumpcyjnych
    df_hcip = get_data_hcip(year_from)

    # Accomodations
    # Nights spent at tourist accommodation establishments
    df_acc = get_data_acc(year_from)

    # Air transport passengers on board
    df_air = get_data_air(year_from)

    # Wskaznik stopy bezrobocia %
    df_jobs = get_data_unemployment(year_from)

    df_return = df_jobs.merge(df_air, how='left', on=['Year', 'Month'])
    df_return = df_return.merge(df_acc, how='left', on=['Year', 'Month'])
    df_return = df_return.merge(df_hcip, how='left', on=['Year', 'Month'])

    return df_return


def get_values_stat():
    values = ['R2', 'RMSE', 'CVS_MEAN']
    return values


def get_values_stat_index():
    values = ['Metoda', 'R2', 'RMSE', 'CVS_MEAN', 'Zbiór treningowy',
              'Parametr']
    return values


def prepare_stats(df):
    columns = get_values_stat()
    for i in range(len(columns)):
        df[columns[i]] = df[columns[i]].astype(float).round(3)
    return df

def prepare_stats_div(df, iterations):
    columns = get_values_stat()
    df = df[columns].copy()
    for i in range(len(columns)):
        df[columns[i]] = df[columns[i]].astype(float).div(iterations).round(3)
    return df

def show_results(df_fusion_selected, df_result_fusion_all, df_result_medical_all,
                 df_result_economy_all, iterations, feature_importances_economy,
                 feature_importances_medical, feature_importances_fusion):
    # Stats
    fusion_data_rows = len(df_fusion_selected)
    print(f'Amount of currently downloaded data (months after fusion): {fusion_data_rows}')
    start_month = df_fusion_selected['Month'].values[0]
    start_year = df_fusion_selected['Year'].values[0]
    print()

    print(f'Start date: {start_month}/{start_year}')
    end_month = df_fusion_selected['Month'].values[len(df_fusion_selected) - 1]
    end_year = df_fusion_selected['Year'].values[len(df_fusion_selected) - 1]
    print(f'End date: {end_month}/{end_year}')

    stats_economy_params = prepare_stats(df_result_economy_all)
    stats_medical_params = prepare_stats(df_result_medical_all)
    stats_fusion_params = prepare_stats(df_result_fusion_all)

    stats_economy_best = stats_economy_params.loc[stats_economy_params.groupby(['Metoda'])['CVS_MEAN'].idxmax()]
    stats_medical_best = stats_medical_params.loc[stats_medical_params.groupby(['Metoda'])['CVS_MEAN'].idxmax()]
    stats_fusion_best = stats_fusion_params.loc[stats_fusion_params.groupby(['Metoda'])['CVS_MEAN'].idxmax()]

    plot_results_by_score(stats_medical=stats_medical_best,
                          stats_economy=stats_economy_best,
                          stats_fusion=stats_fusion_best,
                          iterations=iterations)

    plot_results_by_data(df_stats=stats_medical_best,
                         iterations=iterations,
                         data_name='Dane medyczne',
                         feature_importances=feature_importances_medical)

    plot_results_by_data(df_stats=stats_economy_best,
                         iterations=iterations,
                         data_name='Dane ekonomiczne',
                         feature_importances=feature_importances_economy)

    plot_results_by_data(df_stats=stats_fusion_best,
                         iterations=iterations,
                         data_name='Dane po fuzji',
                         no_features=3,
                         feature_importances=feature_importances_fusion)

    plot_results_by_params(stats_economy_params[stats_economy_params['Metoda'] == 'KNN'],
                           stats_medical_params[stats_medical_params['Metoda'] == 'KNN'],
                           stats_fusion_params[stats_fusion_params['Metoda'] == 'KNN'], iterations, 'liczba sąsiadów')

    plot_results_by_params(stats_economy_params[stats_economy_params['Metoda'] == 'Lasy losowe'],
                           stats_medical_params[stats_medical_params['Metoda'] == 'Lasy losowe'],
                           stats_fusion_params[stats_fusion_params['Metoda'] == 'Lasy losowe'], iterations, 'liczba drzew')
    return


def set_print_options():
    np.set_printoptions(suppress=True,
                        formatter={'float_kind': '{:16.3f}'.format},
                        linewidth=90)
