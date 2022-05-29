import pandas as pd
import numpy as np
from magister_2 import *
from magister_5 import *


def process(year_from=2018, iterations=1):
    # Format print options
    set_print_options()

    # Download: medical data
    df_medical = get_medical_data(year_from)

    # Download: economy data
    df_economy = get_economy_data(year_from)

    # Feature correlation: medical data
    correlation(df=df_medical,
                columns=['Liczba zgonów COVID', 'Nowe zachorowania', 'Nowi zaszczepieni', 'Nowe testy'],
                plot=False,
                data_name='Cechy medyczne')

    # Feature correlation: economy data
    correlation(df=df_economy,
                columns=['HCIP', 'Bezrobotni', 'Transport lotniczy', 'Zakwaterowania'],
                plot=False,
                data_name='Cechy ekonomiczne')

    # Feature selection: medical data
    features_medical, features_medical_importances = \
        get_features_tree(n_features=2,
                          x=df_medical[
                              ['Liczba zgonów COVID', 'Nowe zachorowania', 'Nowi zaszczepieni', 'Nowe testy']],
                          y=df_medical['Liczba zgonów nie COVID'])

    df_medical_selected = df_medical[
        ['Year', 'Month', 'Liczba zgonów nie COVID', features_medical[0], features_medical[1]]]

    # Join non covid deaths and delete empty rows (delete where any value = NaN)
    df_ncd_by_month = get_ncd_by_month(df_medical)
    df_economy = df_economy.merge(df_ncd_by_month, how='left', on=['Year', 'Month'])
    df_economy['Liczba zgonów nie COVID'] = df_economy['Liczba zgonów nie COVID'].fillna(0.0)
    df_economy = df_economy.dropna()

    # Feature selection: economy data
    features_economy, features_economy_importances = \
        get_features_tree(n_features=2,
                          x=df_economy[['HCIP', 'Bezrobotni', 'Transport lotniczy', 'Zakwaterowania']],
                          y=df_economy['Liczba zgonów nie COVID'])

    df_economy_selected = df_economy[['Year', 'Month', 'Liczba zgonów nie COVID', features_economy[0], features_economy[1]]]

    # Regression: economy data
    y = df_economy_selected['Liczba zgonów nie COVID'].values
    x = df_economy_selected[[features_economy[0], features_economy[1]]].values
    df_result_economy_all = \
        regression(n_features=2,
                   x=x,
                   y=y,
                   flg_plot=False,
                   data_name='Dane ekonomiczne',
                   features_names=features_economy,
                   iterations=iterations)

    # Regression: medical data
    y = df_medical_selected['Liczba zgonów nie COVID'].values
    x = df_medical_selected[[features_medical[0], features_medical[1]]].values
    df_result_medical_all = \
        regression(n_features=2,
                   x=x,
                   y=y,
                   flg_plot=False,
                   data_name='Dane medyczne',
                   features_names=features_medical,
                   iterations=iterations)

    # Data fusion
    df_economy_fusion = df_economy_selected[['Year', 'Month', features_economy[0], features_economy[1]]]

    df_medical_fusion = df_medical_selected[
        ['Year', 'Month', 'Liczba zgonów nie COVID', features_medical[0], features_medical[1]]]

    df_fusion = df_economy_fusion.merge(df_medical_fusion, how='left', on=['Year', 'Month'])
    df_fusion = df_fusion.fillna(0.0)

    correlation(df=df_fusion,
                columns=[features_medical[0], features_medical[1], features_economy[0], features_economy[1]],
                plot=True,
                data_name='Cechy po fuzji')

    # Feature selection: data after fusion
    features_fusion, features_fusion_importances \
        = get_features_tree(n_features=3,
                            x=df_fusion[
                                [features_economy[0], features_economy[1], features_medical[0], features_medical[1]]],
                            y=df_fusion['Liczba zgonów nie COVID'])

    df_fusion_selected = df_fusion[
        ['Year', 'Month', 'Liczba zgonów nie COVID', features_fusion[0], features_fusion[1], features_fusion[2]]]

    # Regression: data after fusion
    y = df_fusion_selected['Liczba zgonów nie COVID'].values
    x = df_fusion_selected[[features_fusion[0], features_fusion[1], features_fusion[2]]].values

    df_result_fusion_all = \
        regression(n_features=3,
                   x=x,
                   y=y,
                   flg_plot=False,
                   data_name='Dane po fuzji',
                   features_names=features_fusion,
                   iterations=iterations)

    xmax_economy, xmax_medical, xmax_fusion = show_train_size_plot(iterations=iterations,
                                                                   df_result_economy_all=df_result_economy_all,
                                                                   df_result_fusion_all=df_result_fusion_all,
                                                                   df_result_medical_all=df_result_medical_all)

    df_result_economy_all = df_result_economy_all[df_result_economy_all['Zbiór treningowy'] == xmax_economy / 100]
    df_result_medical_all = df_result_medical_all[df_result_medical_all['Zbiór treningowy'] == xmax_medical / 100]
    df_result_fusion_all = df_result_fusion_all[df_result_fusion_all['Zbiór treningowy'] == xmax_fusion / 100]

    # Stats and plots
    show_results(iterations=iterations,
                 df_result_economy_all=df_result_economy_all,
                 df_result_fusion_all=df_result_fusion_all,
                 df_result_medical_all=df_result_medical_all,
                 df_fusion_selected=df_fusion_selected,
                 feature_importances_fusion=features_fusion_importances,
                 feature_importances_economy=features_economy_importances,
                 feature_importances_medical=features_medical_importances)

process(iterations=5)
