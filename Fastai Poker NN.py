# Poker Hand Dataset


from __future__ import print_function, with_statement, division
import pandas as pd
import fastai
from fastai.metrics import *
from fastai.tabular import *
import torch
from poker import Stringify
                
                
# Baixando os dados e importando-os para um Data Frame
columns = ['N1', 'C1', 'N2', 'C2', 'N3', 'C3', 'N4', 'C4', 'N5', 'C5', 'Jogo']
features = ['N1', 'C1', 'N2', 'C2', 'N3', 'C3', 'N4', 'C4', 'N5', 'C5']
#df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-testing.data', names = columns)
#df.to_csv('Dados\poker_dataset.csv', index = False)
df = pd.read_csv('Dados\poker_dataset.csv')
#df = to_categorical(df)
procs = [Categorify]
path = '\Dados'
test = TabularList.from_df(df.iloc[0:1000].copy(),
                           path='\Dados',
                           cat_names=features)

data = (TabularList.from_df(df, 
                            path='\Dados', 
                            procs = procs,
                            cat_names=features)
                           .split_by_idx(list(range(0,1000)))
                           .label_from_df(cols='Jogo')
                           .add_test(test)
                           .databunch())

# Training
learn = tabular_learner(data, layers=[20,10], metrics=accuracy)
learn
learn.fit(1, 1e-2)








































































