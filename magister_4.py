import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict


# Set extra info upper the bar
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate("{}".format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    rotation='vertical')


def subplot_result_by_score(stats_medical, stats_economy, stats_fusion, score, ax, x):
    bar_width = 0.4
    methods = ['Liniowa', 'KNN', 'Lasy losowe']
    x_arr = []
    height = []
    for i in range(len(methods)):
        x_arr.append(x)
        method = methods[i]
        b1 = ax.bar(x, stats_medical.loc[stats_medical['Metoda'] == method, score],
                    width=bar_width, color='Blue', label='Statystyki dla danych medycznych')
        b2 = ax.bar(x + bar_width, stats_economy.loc[stats_economy['Metoda'] == method, score],
                    width=bar_width, color='Orange', label='Statystyki dla danych ekonomicznych')
        b3 = ax.bar(x + 2 * bar_width, stats_fusion.loc[stats_fusion['Metoda'] == method, score],
                    width=bar_width, color='Green', label='Statystyki dla danych po fuzji')
        x = x + 2

        autolabel(b1, ax)
        autolabel(b2, ax)
        autolabel(b3, ax)
        height.append(stats_medical[score].max())
        height.append(stats_fusion[score].max())
        height.append(stats_economy[score].max())
        ylim = max(height)
        print(ylim)
        if score == 'RMSE':
            ax.set_ylim(0, ylim * 2)
        else:
            ax.set_ylim(0, ylim * 1.4)

    x_arr = np.array(x_arr)
    ax.set_xticks(x_arr + bar_width / 2)
    ax.set_xticklabels(methods)

    if score == 'R2':
        title = 'R\u00B2'
    elif score == 'CVS_MEAN':
        title = 'Średni wynik walidacji krzyżowej'
    elif score == 'RMSE':
        title = 'Średni RMSE'
    else:
        title = score

    ax.set_title(title)
    return ax


def plot_results_by_score(stats_medical, stats_economy, stats_fusion, iterations):
    fig, axs = plt.subplots(ncols=3, figsize=(12, 4))
    x0 = 0.5

    axs[0] = subplot_result_by_score(stats_medical, stats_economy, stats_fusion, score='R2', ax=axs[0], x=x0)
    axs[1] = subplot_result_by_score(stats_medical, stats_economy, stats_fusion, score='CVS_MEAN', ax=axs[1], x=x0)
    axs[2] = subplot_result_by_score(stats_medical, stats_economy, stats_fusion, score='RMSE', ax=axs[2], x=x0)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys())
    fig.tight_layout()
    fig.subplots_adjust(top=0.7)

    train_size_medical = round(stats_medical['Zbiór treningowy'].values[0], 2)
    param_tree_medical = stats_medical.loc[stats_medical['Metoda'] == 'Lasy losowe', 'Parametr'].values[0]
    param_KNN_medical = stats_medical.loc[stats_medical['Metoda'] == 'KNN', 'Parametr'].values[0]

    train_size_economy = round(stats_economy['Zbiór treningowy'].values[0], 2)
    param_tree_economy = stats_economy.loc[stats_economy['Metoda'] == 'Lasy losowe', 'Parametr'].values[0]
    param_KNN_economy = stats_economy.loc[stats_economy['Metoda'] == 'KNN', 'Parametr'].values[0]

    train_size_fusion = round(stats_fusion['Zbiór treningowy'].values[0], 2)
    param_tree_fusion = stats_fusion.loc[stats_fusion['Metoda'] == 'Lasy losowe', 'Parametr'].values[0]
    param_KNN_fusion = stats_fusion.loc[stats_fusion['Metoda'] == 'KNN', 'Parametr'].values[0]

    fig.suptitle(f'Wyniki przed i po fuzji, itercje: {iterations}\
                \nDane ekonomiczne: zbiór treningowy {train_size_economy * 100}%, l. drzew: {param_tree_economy}, l. sąsiadów: {param_KNN_economy}\
                \nDane medyczne: zbiór treningowy {train_size_medical * 100}%, l. drzew: {param_tree_medical}, l. sąsiadów: {param_KNN_medical}\
                \nDane po fuzji: zbiór treningowy {train_size_fusion * 100}%, l. drzew: {param_tree_fusion}, l. sąsiadów: {param_KNN_fusion}')

    fig.text(0.5, 0.04, 'Metody regresji', ha='center', va='center')
    fig.text(0.01, 0.4, 'Wskaźnik', ha='center', va='center', rotation='vertical')
    fig.show()


def subplot_result_by_data(df, ax, x):
    bar_width = 0.4
    methods = ['Liniowa', 'KNN', 'Lasy losowe']
    x_arr = []

    for i in range(len(methods)):
        x_arr.append(x)
        method = methods[i]
        b1 = ax.bar(x, (df.loc[df['Metoda'] == method, 'R2']),
                    width=bar_width, color='Blue', label='R\u00B2')
        b2 = ax.bar(x + bar_width, (df.loc[df['Metoda'] == method, 'CVS_MEAN']),
                    width=bar_width, color='Orange', label='Walidacja krzyżowa (średnia)')

        x = x + 2

        autolabel(b1, ax)
        autolabel(b2, ax)

    x_arr = np.array(x_arr)
    ax.set_xticks(x_arr + bar_width / 2)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.3)
    ax.set_title('R\u00B2, Średni wynik walidacji krzyżowej')

    return ax


def subplot_result_by_data_rmse(df, ax, x):
    bar_width = 0.4
    methods = ['Liniowa', 'KNN', 'Lasy losowe']
    x_arr = []

    for i in range(len(methods)):
        x_arr.append(x)
        method = methods[i]
        b3 = ax.bar(x + bar_width, (df.loc[df['Metoda'] == method, 'RMSE']),
                    width=bar_width, color='Green', label='Średni RMSE')
        x = x + 2
        autolabel(b3, ax)

    x_arr = np.array(x_arr)
    ax.set_xticks(x_arr + bar_width)
    ax.set_xticklabels(methods)
    ax.set_title('Średni RMSE')
    height = df['RMSE'].max()
    ax.set_ylim(0, height * 1.4)
    return ax


def plot_results_by_data(df_stats, iterations, data_name, feature_importances, no_features=2):
    fig, axs = plt.subplots(ncols=3, figsize=(12, 4))
    x0 = 0.5

    axs[0] = subplot_result_by_data(df_stats, ax=axs[0], x=x0)
    axs[1] = subplot_result_by_data_rmse(df_stats, ax=axs[1], x=x0)
    axs[2] = feature_importances.sort_values().plot(kind='barh',
                                                    title=f'Istotności cech; wybrano {no_features}')

    axs[0].set(xlabel='Metody', ylabel='Wartości')
    axs[1].set(xlabel='Metody', ylabel='Wartości')
    axs[2].set(xlabel='Istotności', ylabel='Cechy')

    handles, labels = axs[0].get_legend_handles_labels()
    handles2, labels2 = axs[1].get_legend_handles_labels()
    handles.extend(handles2)
    labels.extend(labels2)

    by_label = OrderedDict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys())
    fig.tight_layout()
    fig.subplots_adjust(top=0.7)
    df_stats = df_stats.reset_index()
    train_size = round(df_stats['Zbiór treningowy'].values[0], 2)
    param_tree = df_stats.loc[df_stats['Metoda'] == 'Lasy losowe', 'Parametr'].values[0]
    param_KNN = df_stats.loc[df_stats['Metoda'] == 'KNN', 'Parametr'].values[0]
    fig.suptitle("\n".join([f'{data_name}, iteracje: {iterations}, zbiór treningowy: {train_size * 100}%,'
                            f' liczba sąsiadów: {param_KNN}, liczba drzew: {param_tree}']))
    plt.autoscale()
    plt.show()


def show_train_size_plot(df_result_medical_all, df_result_economy_all, df_result_fusion_all, iterations):
    fig, ax = plt.subplots()

    df_result_economy_all = df_result_economy_all.groupby('Zbiór treningowy', as_index=False)['CVS_MEAN'].mean()
    ax.plot(df_result_economy_all['Zbiór treningowy'] * 100, df_result_economy_all['CVS_MEAN'], label='Dane ekonomiczne')
    line = fig.gca().get_lines()[0]
    xd = line.get_xdata()
    yd = line.get_ydata()
    xmax_economy = xd[np.argmax(yd)]
    ymax_economy = max(yd)
    ax.annotate('{}'.format(round(xmax_economy, 2)),
                xy=(xmax_economy, ymax_economy),
                xytext=(xmax_economy, ymax_economy), )

    df_result_medical_all = df_result_medical_all.groupby('Zbiór treningowy', as_index=False)['CVS_MEAN'].mean()
    ax.plot(df_result_medical_all['Zbiór treningowy'] * 100, df_result_medical_all['CVS_MEAN'], label='Dane medyczne')
    line = fig.gca().get_lines()[1]
    xd = line.get_xdata()
    yd = line.get_ydata()
    xmax_medical = xd[np.argmax(yd)]
    ymax_medical = max(yd)
    ax.annotate('{}'.format(round(xmax_medical, 2)),
                xy=(xmax_medical, ymax_medical),
                xytext=(xmax_medical, ymax_medical), )

    df_result_fusion_all = df_result_fusion_all.groupby('Zbiór treningowy', as_index=False)['CVS_MEAN'].mean()
    ax.plot(df_result_fusion_all['Zbiór treningowy'] * 100, df_result_fusion_all['CVS_MEAN'], label='Dane po fuzji')
    line = fig.gca().get_lines()[2]
    xd = line.get_xdata()
    yd = line.get_ydata()
    xmax_fusion = xd[np.argmax(yd)]
    ymax_fusion = max(yd)
    ax.annotate('{}'.format(round(xmax_fusion, 2)),
                xy=(xmax_fusion, ymax_fusion),
                xytext=(xmax_fusion, ymax_fusion), )

    ax.plot([xmax_fusion, xmax_economy, xmax_medical], [ymax_fusion, ymax_economy, ymax_medical], '*',
            label='Wartości max')

    fig.legend()
    plt.xlabel("Zbiór treningowy (%)")
    plt.ylabel("Średnia wartość walidacji krzyżowej (dla wszystkich metod i iteracji)")
    plt.suptitle("\n".join([f'Zbiór treningowy VS średnia wartość walidacji krzyżowej, iteracje: {iterations}']))
    plt.autoscale()
    plt.show()
    return round(xmax_economy, 2), round(xmax_medical, 2), round(xmax_fusion, 2)


def plot_results_by_params(df_result_economy_all, df_result_medical_all,
                           df_result_fusion_all, iterations, param_name):
    fig, ax = plt.subplots()
    ax.plot(df_result_economy_all['Parametr'], df_result_economy_all['CVS_MEAN'], label='Dane ekonomiczne')
    line = fig.gca().get_lines()[0]
    xd = line.get_xdata()
    yd = line.get_ydata()
    xmin_economy = xd[np.argmax(yd)]
    ymin_economy = max(yd)
    ax.annotate('{}'.format(round(xmin_economy, 2)),
                xy=(xmin_economy, ymin_economy),
                xytext=(xmin_economy, ymin_economy), )

    ax.plot(df_result_medical_all['Parametr'], (df_result_medical_all['CVS_MEAN']), label='Dane medyczne')
    line = fig.gca().get_lines()[1]
    xd = line.get_xdata()
    yd = line.get_ydata()
    xmin_medical = xd[np.argmax(yd)]
    ymin_medical = max(yd)
    ax.annotate('{}'.format(round(xmin_medical, 2)),
                xy=(xmin_medical, ymin_medical),
                xytext=(xmin_medical, ymin_medical), )

    ax.plot(df_result_fusion_all['Parametr'], (df_result_fusion_all['CVS_MEAN']), label='Dane po fuzji')
    line = fig.gca().get_lines()[2]
    xd = line.get_xdata()
    yd = line.get_ydata()
    xmin_fusion = xd[np.argmax(yd)]
    ymin_fusion = max(yd)
    ax.annotate('{}'.format(round(xmin_fusion, 2)),
                xy=(xmin_fusion, ymin_fusion),
                xytext=(xmin_fusion, ymin_fusion), )

    ax.plot([xmin_fusion, xmin_economy, xmin_medical], [ymin_fusion, ymin_economy, ymin_medical], '*',
            label='Wartości max')

    fig.legend()
    plt.xlabel(f"{param_name}")
    plt.ylabel("Średnia wartość walidacji krzyżowej")
    plt.suptitle("\n".join([f'Średnia wartość walidacji krzyżowej VS parametr {param_name}, iteracje: {iterations}']))
    plt.autoscale()
    plt.show()

    return
