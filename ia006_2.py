import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# PARTE 1

def likelihood (x, mean, variance):
    return np.exp(-np.square(x-mean)/2*np.square(variance))/np.sqrt(2*np.pi)*variance

def MAP (x, mean, variance, priori):
    return likelihood(x, mean, variance)*priori
 
N = list(np.arange(-3,3,0.0001))
likelihood1 = []
likelihood2 = []
threshold = 0
for n in N:
    likelihood1.append(likelihood(n,0,1))
    likelihood2.append(likelihood(n,0,2))
    if likelihood(n,0,2) > likelihood(n,0,1) and threshold == 0:
        threshold = n
print('O valor de x somente é mais verossímil N(0,1) quando {} < x < {}'.format(threshold, -threshold))
plt.plot(N, likelihood1, 'b', N, likelihood2, 'r')
plt.show()

 
N = list(np.arange(-3,3,0.0001))
MAP1 = []
MAP2 = []
threshold = 0
for n in N:
    MAP1.append(MAP(n,0,1,2))
    MAP2.append(MAP(n,0,2,1))
    if MAP(n,0,2,1) > MAP(n,0,1,2) and threshold == 0:
        threshold = n
print('O valor de x somente é mais verossímil N(0,1) para todo x diferente de 0. Em x = 0, a estimativa MAP é igual para as duas distribuições'.format(threshold, -threshold))
plt.plot(N, MAP1, 'b', N, MAP2, 'r')
plt.show()

MAP(0,0,1,2)
MAP(0,0,2,1)

# PARTE 2

two_moons = pd.DataFrame.from_csv(r'C:\Users\Vello\Documents\Dados\two_moons.csv')

# Plotando as classes originais
fig = plt.figure(figsize=(10,10))
ax0 = fig.add_subplot(111)
ax0.scatter(two_moons[two_moons['y'] == 1]['x_1'],two_moons[two_moons['y'] == 1]['x_2'],marker='s',c='grey',edgecolor='black')
ax0.scatter(two_moons[two_moons['y'] == 0]['x_1'],two_moons[two_moons['y'] == 0]['x_2'],marker='^',c='yellow',edgecolor='black')
plt.show()

# 1. Standardize the data
def standerdize (X):
    for i, column in enumerate(X.columns):
        mean = np.mean(X[column])
        std = np.std(X[column])
        for sample in range(len(X)):
            X.iloc[sample,i] = (X.iloc[sample,i]-mean)/std
    return X

X = standerdize(two_moons.iloc[:,0:2])
y = two_moons.iloc[:,2]

    
# 2. Compute the mean vector and the mean vector per class
mean = np.mean(X).values.reshape(2,1)
mean_k = []
for i, c in enumerate(np.unique(y)):
    mean_k.append(np.mean(X.where(y==c),axis=0))
mean_k = np.array(mean_k).T

# 3. Compute the Scatter within and Scatter between matrices
data_SW = []
Nc = []

for i, c, in enumerate(np.unique(y)):
    a = np.array(X.where(y==c).dropna().values-mean_k[:,i].reshape(1,2))
    data_SW.append(np.dot(a.T,a))
    Nc.append(np.sum(y==c))
SW = np.sum(data_SW,axis=0)
SB = np.dot(Nc*np.array(mean_k-mean),np.array(mean_k-mean).T)
    
# 4. Compute the Eigenvalues and Eigenvectors of SW^-1 SB
eigval, eigvec = np.linalg.eig(np.dot(np.linalg.inv(SW),SB))
    
# 5. Select the two largest eigenvalues 
eigen_pairs = [[np.abs(eigval[i]),eigvec[:,i]] for i in range(len(eigval))]
eigen_pairs = sorted(eigen_pairs,key=lambda k: k[0],reverse=True)
w = np.hstack(eigen_pairs[0][1][:,np.newaxis].real) # Select largest
# 6. Transform the data with Y=X*w
X_fisher = X.dot(w)

# Plotando os dados transformados
plt.hist(X_fisher.where(y==0).dropna(),bins=60,alpha=0.5,label='classe 0')
plt.hist(X_fisher.where(y==1).dropna(),bins=60,alpha=0.5,label='classe 1')
plt.legend(loc='upper right')
plt.show()

# Plotando a curva ROC
def ROC (threshold, y_hat, y):
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for sample in range(len(y_hat)):
        if y_hat[sample] >= threshold:
            if y[sample] == 1:
                true_positive += 1
            else:
                false_positive += 1
        else:
            if y[sample] == 0:
                true_negative += 1
            else:
                false_negative += 1
    return false_positive/(false_positive+true_negative), true_positive/(true_positive+false_negative)

true_positive_rate = []
false_positive_rate = []
for threshold in np.arange(-3,3,0.1):
    true_positive_rate.append(ROC(threshold,X_fisher,y)[1])
    false_positive_rate.append(ROC(threshold,X_fisher,y)[0])

plt.plot(false_positive_rate,true_positive_rate)
plt.title('Curva ROC Fisher')
plt.ylabel('Taxa Verdadeiro Positivo')
plt.xlabel('Taxa Falso Positivo')
plt.show()

# Plotando a curva F1
def f1(threshold, y_hat, y):
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for sample in range(len(y_hat)):
        if y_hat[sample] >= threshold:
            if y[sample] == 1:
                true_positive += 1
            else:
                false_positive += 1
        else:
            if y[sample] == 0:
                true_negative += 1
            else:
                false_negative += 1
    recall = true_positive/(true_positive+false_negative)
    precision = true_positive/(true_positive+false_positive)
    return 2*recall*precision/(recall+precision)

f1_list = []
threshold_range = np.arange(-2,2,0.1)
for threshold in threshold_range:
    f1_list.append(f1(threshold,X_fisher,y))
    
plt.plot(list(threshold_range),f1_list)
plt.title('Curva F1 Fisher')
plt.ylabel('F1')
plt.xlabel('Threshold')
plt.show()
    
class RegressaoLogistica:
    def __init__(self, bias = False, lr = 0.01, epochs = 10):
        self.bias = bias
        self.w = None
        self.lr = lr
        self.epochs = epochs
        
    def logistic(self, x):
        return  1 / (1 + np.exp(-np.dot(x, self.w)))
    
    def fit(self, x, y):
        # Adiciona o bias
        if self.bias == True:
            b = np.ones((x.shape[0],x.shape[-1]+1))
            b[:,1:] = x
            x = b
        # Inicia os pesos
        self.w = np.random.normal(0,1,size = x[0].shape)
        # Loop de treinamento
        for _ in range(self.epochs):
            grad = np.zeros(self.w.shape)
            for sample in range(len(grad)):
                grad[sample] +=np.dot((self.logistic(x)-y),x[:,sample])
                
            grad *= self.lr
            self.w -= grad
        
    def coef(self):
        if self.bias == True:
            print('Coeficientes :')
            print(self.w[1:])
            print('Bias :')
            print(self.w[0])
        else :
            print('Coeficientes :')
            print(self.w)
        
    def pred(self, x):
        if self.bias == True:
            b = np.ones((x.shape[0],x.shape[-1]+1))
            b[:,1:] = x
            x = b
        return self.logistic(x)

# Plotando a curva ROC
reglog = RegressaoLogistica(bias = True, lr=0.01, epochs= 1000)
reglog.fit(X,y)

true_positive = []
false_positive = [] 
for threshold in np.arange(0,1,0.1):
    true_positive.append(ROC(threshold,reglog.pred(X),y)[1])
    false_positive.append(ROC(threshold,reglog.pred(X),y)[0])

plt.plot(false_positive,true_positive)
plt.title('Curva ROC Regressão Logistica')
plt.ylabel('Verdadeiro Positivo')
plt.xlabel('Falso Positive')
plt.show()

# Plotando a curva F1
f1_list = []
threshold_range = np.arange(0,1,0.1)
for threshold in threshold_range:
    f1_list.append(f1(threshold,reglog.pred(X),y))
    
plt.plot(list(threshold_range),f1_list)
plt.title('Curva F1 Regressão Logistica')
plt.ylabel('F1')
plt.xlabel('Threshold')
plt.show()
    

# Plotando os dados transformados
plt.hist(pd.Series(reglog.pred(X)).where(y==0).dropna(),bins=60,alpha=0.5,label='classe 0')
plt.hist(pd.Series(reglog.pred(X)).where(y==1).dropna(),bins=60,alpha=0.5,label='classe 1')
plt.legend(loc='upper right')
plt.show()


# PARTE 3

vehicle = pd.read_csv(r'C:\Users\Vello\Documents\Dados\dataset_vehicle.csv')
vehicle.columns

# 1. Standardize the data
x = standerdize(vehicle.iloc[:,:-1])
y = vehicle.iloc[:,-1]

# 2. Split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train = x_train.reset_index()
y_train = y_train.reset_index()
x_test = x_test.reset_index()
y_test = y_test.reset_index()
del x_train['index']
del y_train['index']
del x_test['index']
del y_test['index']


class MultilabelLogisticRegression:
    def __init__(self,x,y):
        self.models = []
        self.predictions = []
        self.combination = []
        self.labels = np.unique(y)
        for i in range(len(self.labels)-1):
            for j in range(i+1,len(self.labels)):
                self.combination.append([self.labels[i],self.labels[j]])
        self.votes = None
    
    def masked_by_label(self,label1,label2,x,y):
        x_copy = x.copy()
        y_copy = y.copy()
        i = 0
        while len(y_copy)<=i:
            if y_copy.iloc[i,0] != label1 and y_copy.iloc[i,0] != label2:
                x_copy.drop(i,inplace=True)
                y_copy.drop(i,inplace=True)
            i += 1
        x_copy = x_copy.reset_index()
        y_copy = y_copy.reset_index()
        del x_copy['index']
        del y_copy['index']
        for i in range(len(y_copy)):
            if y_copy.iloc[i,0] == label1:
                y_copy.iloc[i,0] = 0
            else:
                y_copy.iloc[i,0] = 1
        return x_copy, y_copy

    def fit(self,x,y,bias=True,lr=0.01,epochs=5000):
        for i,(label1,label2) in enumerate(self.combination):
            self.models.append(RegressaoLogistica(bias=bias,lr=lr,epochs=epochs))
            x_masked, y_masked = self.masked_by_label(label1,label2,x,y)
            self.models[i].fit(x_masked,y_masked['Class'])
            print('.')
        
    def pred(self,x):
        self.votes = pd.DataFrame(np.zeros((len(x),len(self.labels))),columns=self.labels)
        for sample in range(len(x)):
            for i, model in enumerate(self.models):
                pred = model.pred(x.iloc[sample,:])
                self.votes[self.combination[i][int(round(pred[0]))]][sample] += abs(pred[0]-0.5)
        for sample in range(len(self.votes)):
            max_score = 0
            max_label = '?'
            for label in self.labels:
                if self.votes[label][sample] > max_score:
                    max_score = a.votes[label][sample]
                    max_label = label
                elif self.votes[label][sample] == max_score:
                    max_label = label + ' ' + max_label
            self.predictions.append(max_label)
        return self.predictions

a = MultilabelLogisticRegression(x,y)
a.fit(x_train,y_train)
a.pred(x_test)

pred = a.pred(x_test)
acc = 0
for i in range(len(y_test)):
    if y_test['Class'][i] == pred[i]:
        acc += 1
print(acc/len(x_test))


