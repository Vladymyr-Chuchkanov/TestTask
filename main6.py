import copy
import random

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

SYMBOL = "BTCUSDT"
START_POPULATION_NUMBER = 100
MAX_GENERATIONS = 100
MUTATION_RATE = 25
CROSSING_RATE = 90
CELLS = 20
MAX_DOWN_PERCENT = 100
MAX_UP_PERCENT = 100
SPLIT_DATA = 100


def outer_target_function(df,strategy):
    income = 0
    step = int(len(df)/SPLIT_DATA)-1
    x = []
    y = []
    for i in range(0,len(df),step):
        df_part = df.iloc[i:i+step]
        temp = target_function(df_part,strategy)
        income +=temp
        x.append(i)
        y.append(temp)
        print(temp)
    plt.hist(y,bins=5)
    plt.show()
    return income/SPLIT_DATA

def target_function(df, strategy0):
    income = 0
    strategy = copy.deepcopy(strategy0)
    strategy = [x for x in strategy if x is not None]
    sorted_strategy = sorted(strategy, key=lambda x: x[1])


    start_price = -1
    bought = []
    end_price = 0
    for i in range(0, len(df)):
        if i == 0:
            start_price = df['Close'][i]
            continue
        if i == len(df)-1:
            end_price = df['Close'][i]
        price = df['Close'][i]
        percent = 100 - (price/start_price)*100
        ind_to_del = []
        if percent > 0:
            for j in range(len(sorted_strategy)):
                el = sorted_strategy[j]
                if el[1] > percent:
                    continue
                else:
                    bought.append([el[0], price, el[2]])
                    ind_to_del.append(j)
            for i in range(len(ind_to_del)):
                sorted_strategy.pop(ind_to_del[len(ind_to_del) - 1 - i])


        ind_to_del = []
        for i in range(len(bought)):
            bought_percent = price/bought[i][1]
            if bought_percent <= 1:
                continue
            if bought_percent >= bought[i][2]/100+1:
                ind_to_del.append(i)
                income += bought[i][0]*bought_percent

        for i in range(len(ind_to_del)):
            bought.pop(ind_to_del[len(ind_to_del)-1-i])
        if len(bought) == 0 and len(sorted_strategy) == 0:
            break
    for el in bought:
        bought_percent = end_price/el[1]
        income += el[0] * bought_percent
    for el in sorted_strategy:
        income+=el[0]

    return income


def generate_start(Start_Number):
    res = []
    for j in range(Start_Number):
        res0 = []
        for i in range(CELLS):
            percent = 100/CELLS/100
            temp = random.randint(1, MAX_DOWN_PERCENT)
            down = temp
            temp = random.randint(1, MAX_DOWN_PERCENT)
            up = temp

            res0.append([percent, down, up])
        res.append(copy.deepcopy(res0))
    return res

def tournament(strategies, incomes):
    res = []
    for i in range(START_POPULATION_NUMBER):
        i1, i2, i3 = 0, 0, 0
        while i1 == i2 or i2 == i3 or i1 == i2:
            i1, i2, i3 = random.randint(0, START_POPULATION_NUMBER - 1), random.randint(0,START_POPULATION_NUMBER - 1), random.randint(0, START_POPULATION_NUMBER - 1)
        if incomes[i1] >= incomes[i2] and incomes[i1] >= incomes[i3]:
            res.append(copy.deepcopy(strategies[i1]))
        elif incomes[i2] >= incomes[i1] and incomes[i2] >= incomes[i3]:
            res.append(copy.deepcopy(strategies[i2]))
        elif incomes[i3] >= incomes[i1] and incomes[i3] >= incomes[i2]:
            res.append(copy.deepcopy(strategies[i3]))
    return res

def breeding(el1,el2):
    point = random.randint(2, len(el1) - 3)
    ch1 = copy.deepcopy(el1[0:point]) + copy.deepcopy(el2[point:len(el2)])
    ch2 = copy.deepcopy(el2[0:point]) + copy.deepcopy(el1[point:len(el1)])
    return ch1, ch2

def mutate(el):
    el2 = copy.deepcopy(el)
    rand = random.randint(0, len(el2) - 1)
    el2[rand][1] = random.randint(1,MAX_DOWN_PERCENT)
    el2[rand][2] = random.randint(1,MAX_DOWN_PERCENT)
    return el2






if __name__ == "__main__":
    df_BTCUSDT = pd.read_csv(SYMBOL + ".csv")  # pd.DataFrame(klines)
    df_BTCUSDT.columns = ["Open time", "Open", "High", "Low", "Close", "Volume", "Close time", "Quote asset volume"
        , "Number of trades", "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"]

    X = df_BTCUSDT['Close time']
    X = pd.to_datetime(X, unit='ms')
    Y = df_BTCUSDT['Close']

    df = pd.DataFrame(data={"Close time": X, "Close": Y})
    df = df.set_index('Close time')
    strategy = [[0.05, 19, 2], [0.05, 2, 1], [0.05, 6, 6], [0.05, 4, 1], [0.05, 2, 10], [0.05, 6, 6], [0.05, 3, 10], [0.05, 8, 5], [0.05, 5, 2], [0.05, 1, 1], [0.05, 2, 13], [0.05, 6, 8], [0.05, 1, 8], [0.05, 14, 1], [0.05, 4, 2], [0.05, 3, 7], [0.05, 9, 9], [0.05, 12, 4], [0.05, 5, 7], [0.05, 1, 8]]
    print(outer_target_function(df,strategy))

    
    """for i in range(len(strategy)):
        strategy[i][1] = strategy[i][1]*0.46
        strategy[i][2] = strategy[i][2]*0.01
    print(target_function(df.iloc[-11000:-10000],strategy))
    
    df_set = df.iloc[-7000:]
    start_price = -1
    x = []
    y = []
    for i in range(len(df_set)):
        if i == 0:
            start_price = df_set['Close'][i]
        price = df_set['Close'][i]
        percent = 100 - (price / start_price * 100)
        y.append(percent)
        x.append(i)
    #plt.plot(x,y)
    #plt.hist(np.array(y),bins=10)
    #df_set.plot()

    #plt.show()"""

 
 
 


    """max_income = 0
    strategies = generate_start(START_POPULATION_NUMBER)
    best_result = []
    for population_index in range(MAX_GENERATIONS):
        print(str(population_index) + " iteration")
        incomes = []
        for strategy in strategies:
            incomes.append(outer_target_function(df,strategy))
        print(str(sum(incomes) / len(incomes)) + " - mean income")
        print(str(max(incomes)) + " - best income")

        if max(incomes) > max_income:
            best_result = strategies[incomes.index(max(incomes))]
            max_income = max(incomes)
        print(best_result)
        strategies = tournament(strategies, incomes)
        after_cross = []
        for el1, el2 in zip(strategies[::2], strategies[1::2]):
            if random.randint(0, 100) < CROSSING_RATE:
                ch1, ch2 = breeding(el1, el2)
                after_cross.append(ch1)
                after_cross.append(ch2)
            else:
                after_cross.append(el1)
                after_cross.append(el2)
        after_mutation = []
        for el in after_cross:
            if random.randint(0, 100) < MUTATION_RATE:
                el1 = mutate(el)
                after_mutation.append(el1)
            else:
                after_mutation.append(el)

        strategies = copy.deepcopy(after_mutation)

    print(best_result)
    print(max_income)"""