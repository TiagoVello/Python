import pandas as pd
from fastai.tabular.transform import TabularProc

def to_categorical(df):
    new_df = pd.DataFrame(columns = df.columns)
    for card_number in [1, 2, 3, 4, 5]:
        card = 'C{}'.format(card_number)
        naipe = 'N{}'.format(card_number)
        new_df[card] = df[card].map({
                1: 'ace',
                2: 'two',
                3: 'three',
                4: 'four',
                5: 'five',
                6: 'six',
                7: 'seven',
                8: 'eight',
                9: 'nine',
                10: 'ten',
                11: 'jack',
                12: 'queen',
                13: 'king',
                })
        new_df[naipe] = df[naipe].map({
                1: 'hearts',
                2: 'spades',
                3: 'diamonds',
                4: 'clubs'
                }) 
    new_df['Jogo'] = df['Jogo'].map({
            0: 'Nothing',
            1: 'One pair',
            2: 'Two pairs',
            3: 'Three of a kind',
            4: 'Straight',
            5: 'Flush',
            6: 'Full house',
            7: 'Four of a kind',
            8: 'Straight flush',
            9: 'Royal flush'
            })
    return new_df
                
class Stringify(TabularProc):
    "Transform the int values to strings."
    def apply_train(self, df:pd.DataFrame):
        for card_number in [1, 2, 3, 4, 5]:
            card = 'C{}'.format(card_number)
            naipe = 'N{}'.format(card_number)
            df[naipe] = df[card].map({
                    1: 'ace',
                    2: 'two',
                    3: 'three',
                    4: 'four',
                    5: 'five',
                    6: 'six',
                    7: 'seven',
                    8: 'eight',
                    9: 'nine',
                    10: 'ten',
                    11: 'jack',
                    12: 'queen',
                    13: 'king',
                    })
            df[naipe] = df[naipe].map({
                    1: 'hearts',
                    2: 'spades',
                    3: 'diamonds',
                    4: 'clubs'
                    }) 

    def apply_test(self, df:pd.DataFrame):
        for card_number in [1, 2, 3, 4, 5]:
            card = 'C{}'.format(card_number)
            naipe = 'N{}'.format(card_number)
            df[naipe] = df[card].map({
                    1: 'ace',
                    2: 'two',
                    3: 'three',
                    4: 'four',
                    5: 'five',
                    6: 'six',
                    7: 'seven',
                    8: 'eight',
                    9: 'nine',
                    10: 'ten',
                    11: 'jack',
                    12: 'queen',
                    13: 'king',
                    })
            df[naipe] = df[naipe].map({
                    1: 'hearts',
                    2: 'spades',
                    3: 'diamonds',
                    4: 'clubs'
                    })         
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                