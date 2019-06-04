import pandas as pd
import numpy as np
from libs import trading_lib as tl

import matplotlib.pyplot as plt

random_days = 12_599


def dji_and_random_hist(make_hist: bool = True, bins: int = 100, log: bool = False) -> (float, float):
    dji_df = pd.read_csv('database/DJIndex.csv')
    perc_cng_index = np.array(dji_df.Close)[1:] / np.array(dji_df.Close)[:-1]
    # perc_cng = (perc_cng - np.mean(perc_cng)) / np.std(perc_cng, ddof=1)

    # Random
    mu, sigma = np.mean(perc_cng_index), np.std(perc_cng_index, ddof=1)
    perc_cng_random = np.random.normal(mu, sigma, 30_000)
    # perc_random = (perc_random - np.mean(perc_random)) / np.std(perc_random, ddof=1)

    if make_hist:
        plt.hist(x=perc_cng_index, bins=bins, density=True, range=(0.80, 1.2), log=log)
        plt.hist(x=perc_cng_random, bins=bins, density=True, range=(0.80, 1.2), alpha=.5, log=log)
        plt.show()

    return mu, sigma


def calc_metrics(capital, years: float = random_days / 252):
    np_capital = np.array(capital)
    cagr = ((np_capital[-1] / np_capital[0]) ** (1 / years) - 1) * 100
    st_dev = np.std(np_capital[1:] / np_capital[:-1] - 1) * np.sqrt(252)
    draw_down = tl.draw_down(list(np_capital))

    return cagr, st_dev, draw_down


def momentum(capital: np.array, perc_random: np.array, buy_hold_text: str):
    df = pd.DataFrame({'capital': capital})
    best_mom = np.array([0, 0, 0])
    for mom in range(1, 25):

        # print(buy_hold_text)

        # Добавим momentum. Он отображает, сменили мы ВЧЕРА на закрытии актив, или нет.
        momentum = capital[22 * mom:] / capital[:-22 * mom]
        momentum = np.insert(momentum, [0] * 22 * mom, 0)
        df[f"momentum_{mom}"] = momentum

        df[f"mom_{mom}_check"] = [np.nan] * len(df)
        for i in range(22 * mom, len(capital), 22):
            df.loc[i + 1, f"mom_{mom}_check"] = 1 if df[f'momentum_{mom}'][i] > 1 else 0
        df[f'mom_{mom}_check'].fillna(method='ffill', inplace=True)
        df[f'mom_{mom}_check'].fillna(0, inplace=True)
        df[f'mom_{mom}_check'].replace({0: False, 1: True}, inplace=True)

        df['perc_cng'] = np.insert(perc_random, 0, 1)
        df[f'mom_{mom}_capital'] = [1] * len(df)
        df.loc[df[f'mom_{mom}_check'].values, f'mom_{mom}_capital'] = df['perc_cng'][df[f'mom_{mom}_check'].values]
        df[f'mom_{mom}_capital'] = np.cumprod(df[f'mom_{mom}_capital'])

        plt.plot(range(1, len(capital) + 1), df[f'mom_{mom}_capital'], label=f'Mom_{mom}')

        cagr, st_dev, draw_down = calc_metrics(df[f'mom_{mom}_capital'])
        best_mom = np.vstack((best_mom, [cagr, st_dev, draw_down]))
        # print(f"Mom: {mom}, CAGR: {cagr}, St_dev: {st_dev}, Down: {draw_down}")
        # print('----------------next mom--------------------------------------')

    best_mom_number = np.where(best_mom == np.max(best_mom[:, 0]))[0][0]
    print(buy_hold_text)
    print(f"best_mom is {best_mom_number} with: {best_mom[best_mom_number]}")
    print('----------------next------------------------------------------')
    # tl.save_csv('database', 'test', df)
    plt.show()


def sma(capital: np.array, perc_random: np.array, buy_hold_text: str):
    df = pd.DataFrame({'capital': capital})
    best_sma = np.array([0, 0, 0])
    for sma in range(25, 425, 25):

        # print(buy_hold_text)

        # Добавим sma. Он отображает, сменили мы ВЧЕРА на закрытии актив, или нет.
        sma_roll = pd.Series(capital).rolling(sma).mean()
        df[f"sma_{sma}"] = capital / sma_roll

        df[f"sma_{sma}_check"] = [np.nan] * len(df)
        for i in range(sma, len(capital), 25):
            df.loc[i + 1, f"sma_{sma}_check"] = 1 if df[f'sma_{sma}'][i] > 1 else 0
        df[f'sma_{sma}_check'].fillna(method='ffill', inplace=True)
        df[f'sma_{sma}_check'].fillna(0, inplace=True)
        df[f'sma_{sma}_check'].replace({0: False, 1: True}, inplace=True)

        df['perc_cng'] = np.insert(perc_random, 0, 1)
        df[f'sma_{sma}_capital'] = [1] * len(df)
        df.loc[df[f'sma_{sma}_check'].values, f'sma_{sma}_capital'] = df['perc_cng'][df[f'sma_{sma}_check'].values]
        df[f'sma_{sma}_capital'] = np.cumprod(df[f'sma_{sma}_capital'])

        plt.plot(range(1, len(capital) + 1), df[f'sma_{sma}_capital'], label=f'SMA_{sma}')

        cagr, st_dev, draw_down = calc_metrics(df[f'sma_{sma}_capital'])
        best_sma = np.vstack((best_sma, [cagr, st_dev, draw_down]))
        # print(f"Sma: {sma}, CAGR: {cagr}, St_dev: {st_dev}, Down: {draw_down}")
        # print('----------------next mom--------------------------------------')

    best_sma_number = np.where(best_sma == np.max(best_sma[:, 0]))[0][0]
    print(buy_hold_text)
    print(f"best_sma is {list(range(0, 425, 25))[best_sma_number]} with: {best_sma[best_sma_number]}")
    print('----------------next------------------------------------------')
    # tl.save_csv('database', 'test', df)
    plt.show()


# Load DJI data
mu, sigma = dji_and_random_hist(make_hist=False)

for i in range(100):
    # np.random.seed(17)
    perc_random = np.random.normal(mu, sigma, random_days)

    # Buy and hold
    capital = np.cumprod(perc_random)
    capital = np.insert(capital, 0, 1)
    cagr, st_dev, draw_down = calc_metrics(capital)
    buy_hold_text = f"BH: CAGR: {cagr}, St_dev: {st_dev}, Down: {draw_down}"

    plt.plot(range(1, len(capital) + 1), capital, label='Buy_hold')
    momentum(capital, perc_random, buy_hold_text)

    plt.plot(range(1, len(capital) + 1), capital, label='Buy_hold')
    sma(capital, perc_random, buy_hold_text)
