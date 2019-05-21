import pandas as pd

class ListadeAtividade ():
    def __init__(self, file):
        data = pd.read_excel(file)
        
        