'''
Each of the original *.gdf files is splitted in several ASCII files:
_s.txt: continuous EEG signals. Each line represents a time point, each column a channel.
_HDR_TRIG.txt: a vector (length #trials) indicating the start of trials (in unit sample.
_HDR_Classlabel.txt: a vector (length #trials) defining the class labels of all trials. The class label is NaN for test trials.
_HDR_ArtifactSelection.txt a vector (length #trials) of boolean values (0/1), a '1' indicating that the respective trials was marked as artifact (see description of the original data).
'''

import pandas
import numpy

def format_row(row):
    nan_list = []
    for _ in range(60):
        nan_list.append(numpy.nan) 
    row = row.split('  ')
    row[-1] = row[-1][:-1]
    del row[0]
    if ' NaN' or 'NaN' or '' in row:
        row = nan_list
    return list(map(float, row))

class TrigFile():
    
    def __init__(self, file_name, mode = 'r'):
        self.file_name = file_name
        self.data = open(file_name, mode)

    def to_list(self):
        data = self.data.readlines()
        data = [0] + list(map(int, data))
        return data
    

class SignalFile():
    
    def __init__(self, file_name, mode = 'r', nan_list_size = 60):
        self.file_name = file_name
        self.data = open(file_name, mode)
        self.columns = []
        for i in range(1, 61):
            self.columns.append(str(i))

    def to_list(self):
       data = self.data.readlines()
       for i in range(len(data)):
           data[i] = format_row(data[i])
       return data



s = SignalFile('k3b_s.txt')
data = s.to_list()
for i in range(len(data)):
    if numpy.nan not in data[i]:
        print(data[i])

    
    

s = open('k3b_s.txt', 'r')
data = s.readlines()


df = pandas.DataFrame(data, columns = columns)
df.head()

# Debug
#strange = []
#for row in range(len(data)):
#    if len(data[row]) > 60:
#        strange.append(row)
#data[strange[1]]