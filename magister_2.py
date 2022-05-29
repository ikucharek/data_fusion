from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import shapiro
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import LeavePOut
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold,StratifiedKFold, cross_val_score
from magister_5 import *


def get_features_tree(n_features, x, y):
    model = RandomForestClassifier(n_jobs=-1, random_state=0)
    model.fit(x, y)
    feat_importances = pd.Series(model.feature_importances_, index=x.columns)
    print('Feature importances:')
    print(feat_importances)
    print()
    feat_importances.nlargest(n_features)
    model = SelectFromModel(model, prefit=True, threshold=-np.inf, max_features=n_features)
    mask = model.get_support()
    list_features = list(x.columns[mask])
    return list_features, feat_importances

def plot_3d(x_test, y_test, X, Y, pred, method_name, features_names, r2, data_name, train_size,
            parameter_name=None, parameter_value=None):
    # Plot model visualization
    plt.style.use('fivethirtyeight')

    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    axes = [ax1, ax2, ax3]

    for ax in axes:
        ax.plot(x_test[:, 0], x_test[:, 1], y_test, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
        ax.scatter(X.flatten(), Y.flatten(), pred, facecolor=(0, 0, 0, 0), s=20,
                   edgecolor='#70b3f0')
        ax.set_xlabel(features_names[0], fontsize=12)
        ax.set_ylabel(features_names[1], fontsize=12)
        ax.set_zlabel('Liczba zgonów nie COVID', fontsize=12)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_zticklabels([])

    ax1.view_init(elev=25, azim=-60)
    ax2.view_init(elev=15, azim=15)
    ax3.view_init(elev=25, azim=60)

    if parameter_name is None:
        fig.suptitle(
            f'{data_name} - {method_name} Wizualizacja modelu ($R\u00B2 = %.2f$); zbiór treningowy: {train_size * 100}proc.' % r2,
            fontsize=10, color='k')
    else:
        fig.suptitle(
            f"{data_name} - {method_name} Wizualizacja modelu ($R\u00B2 = %.2f$); zbiór treningowy: {train_size * 100}proc.; "
            f"{parameter_name}: {parameter_value}" % r2,
        fontsize=10, color='k')

    fig.tight_layout()
    plt.show()


def cross_v_score(x, y, model, n_features):
    kf = KFold(n_splits=2, shuffle=True)
    value = cross_val_score(model, x.reshape(-1, n_features)
                            , y, cv=kf, n_jobs=-1, scoring='r2')
    value = np.around(value, 3)
    print(value.mean())
    return value.mean()


def correlation(df, columns, plot=False, data_name=None):
    pd.set_option('display.max_columns', None)
    df_corr = pd.DataFrame(df, columns=columns)
    stat, p = shapiro(df_corr) #test, if normal distribution
    alpha = 0.05
    if p > alpha:
        corrMatrix = df_corr.corr()
        method = 'pearson'
    else:
        corrMatrix = df_corr.corr(method="spearman")
        method = 'spearman'
    corrMatrix = corrMatrix.round(3)
    if plot == True:
        # fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        maska = np.triu(np.ones_like(corrMatrix, dtype=bool))
        sns.heatmap(data=corrMatrix, mask=maska, annot=True, cmap='BrBG', vmin=-1, vmax=1)
        plt.suptitle(f'{data_name}: Mapa korelacji, metoda {method}', fontsize=10, color='k')
        plt.tight_layout()
        plt.show()

def regression(n_features, x, y, flg_plot, data_name, features_names, iterations):
    df_results_all = pd.DataFrame(columns=get_values_stat_index())

    for j in range(45, 90, 5):
        j = j / 100
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
        df_linear_tmp = pd.DataFrame(columns=get_values_stat_index())
        for i in range(iterations):
            # Linear
            df_linear_new = linear_regression(x, y, x_train, x_test, y_train, y_test, n_features, features_names,
                                              data_name,
                                              train_size=j, flg_plot=False)
            df_linear_tmp[get_values_stat()] = \
                df_linear_tmp[get_values_stat()].add(df_linear_new[get_values_stat()], fill_value=0)
        df_linear_tmp['Metoda'] = df_linear_new['Metoda']
        df_linear_tmp['Zbiór treningowy'] = j
        df_linear_tmp[get_values_stat()] = prepare_stats_div(df_linear_tmp[get_values_stat()], iterations)
        df_results_all = pd.concat([df_results_all, df_linear_tmp], sort=False, ignore_index=True)

    for k in range(1, 11):
        for j in range(45, 90, 5):
            j = j / 100
            x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=j)
            df_KNN_tmp = pd.DataFrame(columns=get_values_stat_index())
            for i in range(iterations):
                # KNN
                df_KNN_new = KNN_regression(x, y, x_train, x_test, y_train, y_test, n_features, features_names,
                                            data_name, param=k,
                                            train_size=j, flg_plot=False)
                df_KNN_tmp[get_values_stat()] = \
                    df_KNN_tmp[get_values_stat()].add(df_KNN_new[get_values_stat()], fill_value=0)
            df_KNN_tmp['Metoda'] = df_KNN_new['Metoda']
            df_KNN_tmp['Zbiór treningowy'] = j
            df_KNN_tmp['Parametr'] = k
            df_KNN_tmp[get_values_stat()] = prepare_stats_div(df_KNN_tmp[get_values_stat()], iterations)
            df_results_all = pd.concat([df_results_all, df_KNN_tmp], sort=False, ignore_index=True)


    for k in range(10, 120, 10):
        for j in range(45, 90, 5):
            j = j / 100
            x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=j)
            df_tree_tmp = pd.DataFrame(columns=get_values_stat_index())
            for i in range(iterations):
                df_tree_new = tree_regression(x, y, x_train, x_test, y_train, y_test, n_features, features_names,
                                            data_name, param=k,
                                            train_size=j, flg_plot=flg_plot)
                df_tree_tmp[get_values_stat()] = \
                    df_tree_tmp[get_values_stat()].add(df_tree_new[get_values_stat()], fill_value=0)
            df_tree_tmp['Metoda'] = df_tree_new['Metoda']
            df_tree_tmp['Zbiór treningowy'] = j
            df_tree_tmp['Parametr'] = k
            df_tree_tmp[get_values_stat()] = prepare_stats_div(df_tree_tmp[get_values_stat()], iterations)
            df_results_all = pd.concat([df_results_all, df_tree_tmp], sort=False, ignore_index=True)

    return df_results_all


def create_meshgrid(n_features, x):
    if n_features == 2:
        x_plot_1 = np.linspace(start=x.reshape((-1, n_features))[:, 0].min(),
                               stop=x.reshape((-1, n_features))[:, 0].max(), num=50)
        x_plot_2 = np.linspace(start=x.reshape((-1, n_features))[:, 1].min(),
                               stop=x.reshape((-1, n_features))[:, 1].max(), num=50)
        X, Y = np.meshgrid(x_plot_1, x_plot_2)
        x_plot = np.array([X.flatten(), Y.flatten()]).T

    elif n_features == 3:
        x_plot_1 = np.linspace(start=x.reshape((-1, n_features))[:, 0].min(),
                               stop=x.reshape((-1, n_features))[:, 0].max(), num=50)
        x_plot_2 = np.linspace(start=x.reshape((-1, n_features))[:, 1].min(),
                               stop=x.reshape((-1, n_features))[:, 1].max(), num=50)
        x_plot_3 = np.linspace(start=x.reshape((-1, n_features))[:, 2].min(),
                               stop=x.reshape((-1, n_features))[:, 2].max(), num=50)
        X, Y, Z = np.meshgrid(x_plot_1, x_plot_2, x_plot_3)
        x_plot = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T

    return X, Y, x_plot


def create_stats(method_name, param, train_size, r2, rmse, cvs_mean):
    data = {'Metoda': [method_name],
            'Parametr': [param],
            'Zbiór treningowy': train_size,
            'R2': [r2],
            'RMSE': [rmse],
            'CVS_MEAN': [cvs_mean]}
    df = pd.DataFrame(data)
    return df


def linear_regression(x, y, x_train, x_test, y_train, y_test, n_features, features_names, data_name, train_size,
                      flg_plot):
    model = LinearRegression()
    model.fit(x_train.reshape((-1, n_features)), y_train)
    mse = mean_squared_error(y_test, model.predict(x_test.reshape((-1, n_features))))
    rmse = round(math.sqrt(mse)/len(y_test), 3)
    cvs_mean = cross_v_score(x, y, model, n_features)

    X, Y, x_plot = create_meshgrid(n_features, x)

    pred = model.predict(x_plot)
    r2 = model.score(x_train.reshape((-1, n_features)), y_train)

    if flg_plot is True and n_features == 2:
        plot_3d(x_test=x_test, y_test=y_test, X=X, Y=Y, pred=pred, method_name='Regresja liniowa',
                features_names=features_names, r2=r2,
                data_name=data_name, train_size=train_size)

    df_stats = create_stats(method_name='Liniowa', param=None, train_size=train_size, r2=r2, rmse=rmse, cvs_mean=cvs_mean)

    return df_stats


def KNN_regression(x, y, x_train, x_test, y_train, y_test, n_features, features_names, data_name, train_size, param,
                   flg_plot):
    model = KNeighborsRegressor(n_neighbors=param)
    model.fit(x_train.reshape((-1, n_features)), y_train)
    mse = mean_squared_error(y_test, model.predict(x_test.reshape((-1, n_features))))
    rmse = round(math.sqrt(mse)/len(y_test), 3)
    cvs_mean = cross_v_score(x, y, model, n_features)

    X, Y, x_plot = create_meshgrid(n_features, x)

    pred = model.predict(x_plot)
    r2 = model.score(x_train.reshape((-1, n_features)), y_train)

    if flg_plot is True and n_features == 2:
        plot_3d(x_test=x_test, y_test=y_test, X=X, Y=Y, pred=pred, method_name='Regresja kNN',
                features_names=features_names, r2=r2,
                data_name=data_name, train_size=train_size, parameter_name='liczba sąsiadów', parameter_value=param)

    df_stats = create_stats(method_name='KNN', param=param, train_size=train_size, r2=r2, rmse=rmse, cvs_mean=cvs_mean)

    return df_stats


def tree_regression(x, y, x_train, x_test, y_train, y_test, n_features, features_names, data_name, train_size, param,
                    flg_plot):
    model = RandomForestRegressor(n_estimators=param)
    model.fit(x_train.reshape((-1, n_features)), y_train)
    mse = mean_squared_error(y_test, model.predict(x_test.reshape((-1, n_features))))
    rmse = round(math.sqrt(mse)/len(y_test), 3)
    cvs_mean = cross_v_score(x, y, model, n_features)

    X, Y, x_plot = create_meshgrid(n_features, x)

    pred = model.predict(x_plot)
    r2 = model.score(x_train.reshape((-1, n_features)), y_train)

    if flg_plot is True and n_features == 2:
        plot_3d(x_test=x_test, y_test=y_test, X=X, Y=Y, pred=pred, method_name='Regresja lasy losowe',
                features_names=features_names, r2=r2,
                data_name=data_name, train_size=train_size, parameter_name='liczba drzew', parameter_value=param)

    df_stats = create_stats(method_name='Lasy losowe', param=param, train_size=train_size, r2=r2, rmse=rmse,
                            cvs_mean=cvs_mean)
    return df_stats
