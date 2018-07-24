from random import randint
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

Yield = []
DrawDown = []
Number = []
Zero = 0

Table = pd.DataFrame({})
while Zero < 1000:
    Tickers = 62
    Aim = 175
    Low = 800
    High = 1500
    Random = []

    while Tickers > 0:
        First = randint(-Low, High)
        Random.append(First)
        Capit = (First / 10000 + 1)

        if First > 0:
            Second = randint(-Low, 0)
        elif First < 0:
            Second = randint(0, High)
        Random.append(Second)
        Capit = (Capit * (Second/10000) + Capit)

        if First+Second < 0:
            Min = -Low+Aim+abs(First+Second)
            Max = abs(First+Second)+Aim+High
            Third = randint(Min, Max)
        elif First+Second > 0:
            Min = -Low+Aim-abs(First+Second)
            Max = -abs(First+Second)+Aim+High
            Third = randint(Min, Max)
        Random.append(Third)
        Capit = (Capit * (Third / 10000) + Capit)

        Fourth = round((1.0175/Capit-1)*10000,0)
        Random.append(Fourth)

        Tickers -= 1

    #Миксуем наши полученные значения
    np.random.shuffle(Random)
    # for i in range(0,len(Random)):
        # Random[i] = Random[randint(0,len(Random)-1)]

    # Random[:] = [i + randint(-150, 200) for i in Random]
    Random[:] = [i / 10000 for i in Random]

    Capital = []
    Count = []
    High = []
    Down = []
    for i in range(0,len(Random)):
        if i == 0:
            Capital.append(10000)
            Count.append(1)
            High.append(Capital[-1])
        else:
            Capital.append(Random[i-1]*Capital[-1]+Capital[-1])
            Count.append(Count[-1]+1)
            if Capital[-1] > High[-1]:
                High.append(Capital[-1])
                Down.append(Capital[-1]/High[-1])
            else:
                High.append(High[-1])
                Down.append(Capital[-1]/High[-1])

    Yield.append(Capital[-1]/Capital[0])
    DrawDown.append(round((min(Down)-1)*100, 1))
    Number.append(Zero)
    Zero += 1

    Table[str(Zero)] = Capital
    plt.plot(Count, Table[str(Zero)], alpha=0.8)
    # #Рисуем график
    # _, ax = plt.subplots()
    # ax.plot(Count, Capital, lw=2, color='#539caf', alpha=1.0)
    # ax.set_title("250 сделок Дей-Трейдера за год")
    # ax.set_xlabel("DrawDown "+str(round((1-min(Down))*100,1))+"%")
    # ax.set_ylabel("Капитал")
    # plt.show(block=True)

    # RandomEarnings = pd.DataFrame({"Random": Random})
    # RandomEarnings.to_csv("RandomEarnings.csv")

print(len(Number))

_, ax = plt.subplots()
# plt.scatter(Count, Table["One"], marker="o", s=10, alpha=0.8)
ax.set_xlabel("Down")
ax.set_ylabel("Gain")
plt.show()
