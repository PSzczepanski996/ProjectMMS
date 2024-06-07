import numpy as np
import pandas as pd
from scipy import stats

cached_files = []
filenames = [
    'data/OnlineNewsPopularity.csv',
]
for item in filenames:
    df = pd.read_csv(item, index_col=None, header=0)
    cached_files.append(df)

frame = pd.concat(cached_files, axis=0, ignore_index=True)
# print(frame)


def get_data_info(frame):
    print(f'Tabela: \n{frame.head()}')
    print(10*f'-')
    print(f'Opis: \n{frame.describe()}')
    print(10 * f'-')
    print(f'Informacje: \n{frame.info()}')
    print(10 * f'-')
    print(f'Braki danych: \n{frame.isnull().sum()}')
    print(10 * f'-')
    print(f'Kolumny: \n{frame.columns}')
    print(10 * f'-')
    print(f'Typy danych: \n{frame.dtypes}')
    print(10 * f'-')
    print(f'Liczby wierszy: \n' + str(len(frame[' timedelta'])))
    print(10 * f'-')


# get_data_info(frame)


def generate_statistics(df, column_name):
    mean = round(df[column_name].mean(), 2)
    median = round(df[column_name].median(), 2)
    std = round(df[column_name].std(), 2)
    var = round(df[column_name].var(), 2)
    skewness = round(df[column_name].skew(), 2)
    kuri = round(df[column_name].kurtosis(), 2)
    sigma = round(np.sum((df[column_name] > mean - 3 * std) & (df[column_name] < mean + 3 * std)) / len(df) * 100, 2)

    skewness_description = 'Brak skośności'
    if skewness > 0:
        skewness_description = 'Prawostronna'
    elif skewness < 0:
        skewness_description = 'Lewostronna'

    kuri_excess = kuri - 3
    kuri_description = 'Rozkład mezokurtyczny'
    if kuri > 0:
        kuri_description = 'Rozkład leptykurtyczny'
    elif kuri < 0:
        kuri_description = 'Rozkład plakurtyczny'

    print(f'Średnia dla {column_name}: {mean}')
    print(f'Mediana dla {column_name}: {median}')
    print(f'Odchylenie standardowe dla {column_name}: {std}')
    print(f'Wariancja dla {column_name}: {var}')
    print(f'Skośność dla {column_name}: {skewness} ', f'\t | \t Opis: {skewness_description} skośność')
    print(f'Kurtoza dla {column_name}: {kuri}', f'\t | \t Opis: {kuri_description} ',
          f'\t | \t Nadmiar kurtozy: {kuri_excess}')
    print(f'Procent wartości mieszczących się w przedziale 3 sigma dla {column_name}')


def generate_and_compare(df, column_name, number_of_intervals):
    sorted_values = df[column_name].sort_values()
    min_values = sorted_values.min()
    max_values = sorted_values.max()

    print(f'Min wartość dla {column_name}: {min_values}')
    print(f'Max wartość dla {column_name}: {max_values}')

    interval_width = (max_values - min_values) / number_of_intervals

    intervals = [min_values + i * interval_width for i in range(number_of_intervals + 1)]

    print(10 * f'=')

    sorted_values = sorted_values.to_frame()

    sorted_values['interval'] = pd.cut(sorted_values[column_name], bins=intervals,
                                       labels=[f'interval_{i}' for i in range(number_of_intervals)])

    print(sorted_values['interval'].value_counts())
    most_frequent_interval = sorted_values['interval'].value_counts().idxmax()
    TEMP_df = sorted_values[sorted_values['interval'] == most_frequent_interval]

    # print(TEMP_df)
    print(10 * f'=')
    print(f"Wartosci statystyczne dla {number_of_intervals} przedziałów dla {column_name}")
    generate_statistics(TEMP_df, column_name)
    print(10 * f'=')


def goodness_of_fit(df, column_name, alpha):

    print('Test zgodności z rozkładem normalnym:')
    data = df[column_name].values
    normality_test = stats.normaltest(data)
    print(f'Statystyka testowa: {normality_test.statistic}')
    print(f'P-value: {normality_test.pvalue}')
    if normality_test.pvalue < alpha:
        print(f"Dla poziomu istotności {alpha}, odrzucamy hipotezę zerową o normalności rozkładu.")
    else:
        print(
            f'Dla poziomu istotności {alpha}, nie ma podstaw do odrzucenia hipotezy zerowej o normalności rozkładu.'
        )

    # Testowanie zgodności za pomocą testu Kołmogorowa-Smirnowa
    print('\nTest Kołmogorowa-Smirnowa:')
    ks_test = stats.kstest(data, 'norm')
    print(f'Statystyka testowa: {ks_test.statistic}')
    print(f'P-value: {ks_test.pvalue}')
    if ks_test.pvalue < alpha:
        print(
            f'Dla poziomu istotności {alpha}, odrzucamy hipotezę zerową o zgodności rozkładu z rozkładem normalnym.',
        )
    else:
        print(
            f'Dla poziomu istotności {alpha}, nie ma podstaw do odrzucenia hipotezy zerowej o zgodności rozkładu z rozkładem normalnym.'
        )

    # Testowanie zgodności za pomocą testu chi-kwadrat
    print('\nTest chi-kwadrat:')
    chi2, p_value = stats.chisquare(data)
    print(f'Statystyka testowa: {chi2}')
    print(f'P-value: {p_value}')
    if p_value < alpha:
        print(
            f'Dla poziomu istotności {alpha}, odrzucamy hipotezę zerową o zgodności rozkładu.'
        )
    else:
        print(
            f'Dla poziomu istotności {alpha}, nie ma podstaw do odrzucenia hipotezy zerowej o zgodności rozkładu.'
        )


attribute1 = ' timedelta'
attribute2 = ' shares'
const_n = 10
alpha = 0.02

# Wywołanie funkcji dla obu cech:
for item in [attribute2]:
    # generate_statistics(frame, item)
    # generate_and_compare(frame, item, const_n)
    goodness_of_fit(frame, item, alpha)