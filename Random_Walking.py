import pandas as pd
import numpy as np
from libs import trading_lib as tl

import matplotlib.pyplot as plt

random_days = 12_599


def momentum(capital: np.array, perc_random: np.array, buy_hold_text: str):
    df = pd.DataFrame({'capital': capital})
    best_mom = np.array([0, 0, 0])
    for mom in range(1, 25):

        # print(buy_hold_text)

        # Добавим momentum. Он отображает, сменили мы ВЧЕРА на закрытии актив, или нет.
        momentum = capital[22 * mom:] / capital[:-22 * mom]
        momentum = np.insert(momentum, [0] * 22 * mom, 0)
        df[f"momentum_{mom}"] =  momentum

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
        np_capital = np.array(df[f'mom_{mom}_capital'])
        cagr = ((np_capital[-1] / np_capital[0]) ** (1 / years) - 1) * 100
        st_dev = np.std(np_capital[1:] / np_capital[:-1] - 1) * np.sqrt(252)
        draw_down = tl.draw_down(list(np_capital))
        # print(f"Mom: {mom}, CAGR: {cagr}, St_dev: {st_dev}, Down: {draw_down}")
        # print('----------------next mom--------------------------------------')
        best_mom = np.vstack((best_mom, [cagr, st_dev, draw_down]))

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

        # tl.save_csv('database', 'test', df)

        plt.plot(range(1, len(capital) + 1), df[f'sma_{sma}_capital'], label=f'SMA_{sma}')
        np_capital = np.array(df[f'sma_{sma}_capital'])
        cagr = ((np_capital[-1] / np_capital[0]) ** (1 / years) - 1) * 100
        st_dev = np.std(np_capital[1:] / np_capital[:-1] - 1) * np.sqrt(252)
        draw_down = tl.draw_down(list(np_capital))
        # print(f"Sma: {sma}, CAGR: {cagr}, St_dev: {st_dev}, Down: {draw_down}")
        # print('----------------next mom--------------------------------------')
        best_sma = np.vstack((best_sma, [cagr, st_dev, draw_down]))

    best_sma_number = np.where(best_sma == np.max(best_sma[:, 0]))[0][0]
    print(buy_hold_text)
    print(f"best_sma is {list(range(0, 425, 25))[best_sma_number]} with: {best_sma[best_sma_number]}")
    print('----------------next------------------------------------------')
    plt.show()


# Load DJI data
dji_df = pd.read_csv('database/DJIndex.csv')
perc_cng = np.array(dji_df.Close)[1:] / np.array(dji_df.Close)[:-1]
# perc_cng = (perc_cng - np.mean(perc_cng)) / np.std(perc_cng, ddof=1)
# plt.hist(x=perc_cng, bins=100, density=True, range=(0.80, 1.2), log=True)

# Random
mu, sigma = np.mean(perc_cng), np.std(perc_cng, ddof=1)
# perc_random = np.random.normal(mu, sigma, 30_000)
# perc_random = (perc_random - np.mean(perc_random)) / np.std(perc_random, ddof=1)
# plt.hist(x=perc_random, bins=100, density=True, range=(0.80, 1.2), alpha=.5, log=True)
# plt.show()

for i in range(100):
    # np.random.seed(17)
    perc_random = np.random.normal(mu, sigma, random_days)

    # Buy and hold
    capital = np.cumprod(perc_random)
    capital = np.insert(capital, 0, 1)
    plt.plot(range(1, len(capital) + 1), capital, label='Buy_hold')

    years = random_days / 252
    cagr = ((capital[-1] / capital[0]) ** (1 / years) - 1) * 100
    st_dev = np.std(capital[1:] / capital[:-1] - 1) * np.sqrt(252)
    draw_down = tl.draw_down(list(capital))
    buy_hold_text = f"Mom: {0}, CAGR: {cagr}, St_dev: {st_dev}, Down: {draw_down}"

    momentum(capital, perc_random, buy_hold_text)

    plt.plot(range(1, len(capital) + 1), capital, label='Buy_hold')
    sma(capital, perc_random, buy_hold_text)
