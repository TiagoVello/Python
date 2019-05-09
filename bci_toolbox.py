# =============================================================================
# 
# Each of the original *.gdf files is splitted in several ASCII files:
# _s.txt: continuous EEG signals. Each line represents a time point, each column a channel.
# _HDR_TRIG.txt: a vector (length #trials) indicating the start of trials (in unit sample.
# _HDR_Classlabel.txt: a vector (length #trials) defining the class labels of all trials. The class label is NaN for test trials.
# _HDR_ArtifactSelection.txt a vector (length #trials) of boolean values (0/1), a '1' indicating that the respective trials was marked as artifact (see description of the original data).
# 
# =============================================================================

import pandas
import numpy
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


# Global variables
nan_list = []
for _ in range(60):
    nan_list.append(numpy.nan)
    
def format_row(row):
    row = row.split('  ')
    row[-1] = row[-1][:-1]
    row = row[1:]
    if ' NaN' in row:
        row = nan_list
    row = list(map(float, row))
    return row

class TrigFile():
    
    def __init__(self, file_name, mode = 'r'):
        self.file_name = file_name
        self.data = open(file_name, mode)

    def to_list(self):
        data = self.data.readlines()
        data = [0] + list(map(int, data))
        return data
    

class SignalFile():
    
    def __init__(self, file_name, mode = 'r'):
        self.data = open(file_name, mode)
        self.columns = []
        for i in range(1, 61):
            self.columns.append(str(i))

    def to_list(self):
       data = self.data.readlines()
       for i in range(len(data)):
           data[i] = format_row(data[i])
       return data
    
    
def label_file(file_name):
    yf = open(file_name, 'r')
    yl = yf.readlines()
    for label in range(len(yl)):
        yl[label] = yl[label][:-2]
        yl[label] = float(yl[label])
    return yl

y = label_file('k3b_HDR_Classlabel.txt')   
    
t = TrigFile('k3b_HDR_TRIG.txt')
t = t.to_list()


s = SignalFile('k3b_s.txt')
data = s.to_list()
df = pandas.DataFrame(data, columns = s.columns) 

def show_trial(fs = 50, trial = 0, only_c = False):
    label_int = int(y[trial])
    if label_int == 1:
        label = 'Left Hand'
    elif label_int == 2:
        label = 'Right Hand'
    elif label_int == 3:
        label = 'Foot'
    elif label_int == 4:
        label = 'Tongue'
    print('Channels for label: {}'.format(label))
    if only_c:
        for i in [28,31,34]:
            if i == 28:
                title = 'C3'
            elif i == 31:
                title = 'Cz'
            elif i == 34:
                title = 'C4'
            test = df.iloc[t[trial]:t[trial+1],i].values
            f ,time ,Sxx = spectrogram(test, fs, return_onesided=False)
            plt.pcolormesh(time, numpy.fft.fftshift(f), numpy.fft.fftshift(Sxx, axes=0))
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.title(title)
            plt.show()
    else:
        for i in range(60):
            test = df.iloc[t[trial]:t[trial+1],i].values
            f ,time ,Sxx = spectrogram(test, fs, return_onesided=False)
            plt.pcolormesh(time, numpy.fft.fftshift(f), numpy.fft.fftshift(Sxx, axes=0))
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.title('{}'.format(i))
            plt.show()
            
show_trial(trial = 8, only_c = False)


save_test = df.iloc[t[0]:t[1],31].values
f ,time ,Sxx = spectrogram(save_test, 50, return_onesided=False)
plt.pcolormesh(time, numpy.fft.fftshift(f), numpy.fft.fftshift(Sxx, axes=0))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('save_test')
plt.show()
plt.savefig('save_teste')

    

y = label_file('k3b_HDR_Classlabel.txt')
df_label = pandas.DataFrame({'y':y})
len(df_label)

#Debug
# =============================================================================
# strange = []
# for row in range(len(data)):
#     if len(data[row]) > 60:
#         strange.append(row)
# data[strange[1]]
# =============================================================================
