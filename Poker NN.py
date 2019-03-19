
# =============================================================================
# 1. Title: Poker Hand Dataset 
# 
# 2. Source Information
# 
# a) Creators:
# 
# 	Robert Cattral (cattral@gmail.com)
# 
# 	Franz Oppacher (oppacher@scs.carleton.ca)
# 	Carleton University, Department of Computer Science
# 	Intelligent Systems Research Unit
# 	1125 Colonel By Drive, Ottawa, Ontario, Canada, K1S5B6
# 
# c) Date of release: Jan 2007
#  
# 3. Past Usage:
#     1. R. Cattral, F. Oppacher, D. Deugo. Evolutionary Data Mining
#        with Automatic Rule Generalization. Recent Advances in Computers,
#        Computing and Communications, pp.296-300, WSEAS Press, 2002.
#        - Note: This was a slightly different dataset that had more
#                classes, and was considerably more difficult.
# 
#     - Predictive attribute: Poker Hand (labeled �class�)
#     - Found to be a challenging dataset for classification algorithms
#     - Relational learners have an advantage for some classes
#     - The ability to learn high level constructs has an advantage
# 
# 4. Relevant Information:
#      Each record is an example of a hand consisting of five playing
#      cards drawn from a standard deck of 52. Each card is described
#      using two attributes (suit and rank), for a total of 10 predictive
#      attributes. There is one Class attribute that describes the
#      �Poker Hand�. The order of cards is important, which is why there
#      are 480 possible Royal Flush hands as compared to 4 (one for each
#      suit � explained in more detail below).
# 
# 5. Number of Instances: 25010 training, 1,000,000 testing
# 
# 6. Number of Attributes: 10 predictive attributes, 1 goal attribute
# 
# 7. Attribute Information:
#    1) S1 �Suit of card #1�
#       Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}
# 
#    2) C1 �Rank of card #1�
#       Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)
# 
#    3) S2 �Suit of card #2�
#       Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}
# 
#    4) C2 �Rank of card #2�
#       Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)
# 
#    5) S3 �Suit of card #3�
#       Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}
# 
#    6) C3 �Rank of card #3�
#       Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)
# 
#    7) S4 �Suit of card #4�
#       Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}
# 
#    8) C4 �Rank of card #4�
#       Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)
# 
#    9) S5 �Suit of card #5�
#       Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}
# 
#    10) C5 �Rank of card 5�
#       Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)
# 
#    11) CLASS �Poker Hand�
#       Ordinal (0-9)
# 
#       0: Nothing in hand; not a recognized poker hand 
#       1: One pair; one pair of equal ranks within five cards
#       2: Two pairs; two pairs of equal ranks within five cards
#       3: Three of a kind; three equal ranks within five cards
#       4: Straight; five cards, sequentially ranked with no gaps
#       5: Flush; five cards with the same suit
#       6: Full house; pair + different rank three of a kind
#       7: Four of a kind; four equal ranks within five cards
#       8: Straight flush; straight + flush
#       9: Royal flush; {Ace, King, Queen, Jack, Ten} + flush
# 
# 
# 8. Missing Attribute Values: None
# 
# 9. Class Distribution:
# 
#       The first percentage in parenthesis is the representation
#       within the training set. The second is the probability in the full domain.
# 
#       Training set:
# 
#       0: Nothing in hand, 12493 instances (49.95202% / 50.117739%)
#       1: One pair, 10599 instances, (42.37905% / 42.256903%)
#       2: Two pairs, 1206 instances, (4.82207% / 4.753902%)
#       3: Three of a kind, 513 instances, (2.05118% / 2.112845%)
#       4: Straight, 93 instances, (0.37185% / 0.392465%)
#       5: Flush, 54 instances, (0.21591% / 0.19654%)
#       6: Full house, 36 instances, (0.14394% / 0.144058%)
#       7: Four of a kind, 6 instances, (0.02399% / 0.02401%)
#       8: Straight flush, 5 instances, (0.01999% / 0.001385%)
#       9: Royal flush, 5 instances, (0.01999% / 0.000154%)
# 
#       The Straight flush and Royal flush hands are not as representative of
#       the true domain because they have been over-sampled. The Straight flush
#       is 14.43 times more likely to occur in the training set, while the
#       Royal flush is 129.82 times more likely.
# 
#       Total of 25010 instances in a domain of 311,875,200.
# 
#       Testing set:
# 
# 	The value inside parenthesis indicates the representation within the test
#       set as compared to the entire domain. 1.0 would be perfect 
#       representation,while <1.0 are under-represented and >1.0 are 
#       over-represented.
# 
#       0: Nothing in hand, 501209 instances,(1.000063)
#       1: One pair, 422498 instances,(0.999832)
#       2: Two pairs, 47622 instances, (1.001746)
#       3: Three of a kind, 21121 instances, (0.999647)
#       4: Straight, 3885 instances, (0.989897)
#       5: Flush, 1996 instances, (1.015569)
#       6: Full house, 1424 instances, (0.988491)
#       7: Four of a kind, 230 instances, (0.957934)
#       8: Straight flush, 12 instances, (0.866426)
#       9: Royal flush, 3 instances, (1.948052)
# 
#       Total of one million instances in a domain of 311,875,200.
# 
# 
# 10. Statistics
# 
#       Poker Hand       # of hands	Probability	# of combinations
#       Royal Flush      4		0.00000154	480
#       Straight Flush   36		0.00001385	4320
#       Four of a kind   624		0.0002401	74880
#       Full house       3744		0.00144058	449280
#       Flush            5108		0.0019654	612960
#       Straight         10200		0.00392464	1224000
#       Three of a kind  54912		0.02112845	6589440
#       Two pairs        123552		0.04753902	14826240
#       One pair         1098240	0.42256903	131788800
#       Nothing          1302540	0.50117739	156304800
# 
#       Total            2598960	1.0		311875200
# 
#       The number of combinations represents the number of instances in the 
#       entire domain.
# =============================================================================

from __future__ import print_function, with_statement, division
import pandas as pd
import random
import time
import torch
import torch.nn
import itertools
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ExponentialLR

# Baixando os dados e importando-os para um Data Frame
columns = ['N1', 'C1', 'N2', 'C2', 'N3', 'C3', 'N4', 'C4', 'N5', 'C5',
             'Jogo']
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-testing.data',
                 names = columns)
df.to_csv('poker_dataset.csv')
X = df.iloc[:, :10]
y = df.iloc[:, 10]

# Criando um dataset que possui os dados de treinamento e de teste
class PokerDataset(Dataset):
    def __init__(self, X, y, random_state=None, transforms=None, mode=None):

        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                           test_size=0.15,
                                           random_state=random_state)
        if mode == 'train':
            self.X = X_train
            self.y = y_train
        if mode == 'test':
            self.X = X_test
            self.y = y_test
        if mode == None:
            self.X = X
            self.y = y
        self.mode = mode
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X.iloc[idx]
        y = self.y.iloc[idx]
        sample = {'X': X, 'y': y}
        if self.transforms:
            for transform in self.transforms:
                sample = transform(sample)
        return sample

# Transformar os dados para que a ordem das cartas na mão não altere o jogo
class ShuffleHand(object):   
    def __call__(self, sample):
        original_order = [1, 2, 3, 4, 5]
        X, y = sample['X'], sample['y']
        shuffled_order = random.sample(original_order, 5)
        N1 = X['N{}'.format(shuffled_order[0])]
        N2 = X['N{}'.format(shuffled_order[1])]
        N3 = X['N{}'.format(shuffled_order[2])]
        N4 = X['N{}'.format(shuffled_order[3])]
        N5 = X['N{}'.format(shuffled_order[4])]
        C1 = X['C{}'.format(shuffled_order[0])]
        C2 = X['C{}'.format(shuffled_order[1])]
        C3 = X['C{}'.format(shuffled_order[2])]
        C4 = X['C{}'.format(shuffled_order[3])]
        C5 = X['C{}'.format(shuffled_order[4])]
        return {'X': [N1, C1, N2, C2, N3, C3, N4, C4, N5, C5], 'y': y}

class ToTensor(object):
    def __call__(self, sample):
        X = torch.tensor(sample['X'], dtype = torch.float)
        y = torch.tensor(sample['y'], dtype = torch.float)
        return {'X': X, 'y': y }

class IsPair(object):
    def __call__(self, sample):
        X = sample['X']
        y = sample['y']
        X = [X[1], X[3], X[5], X[7], X[9]]
        pair_flag = 0
        for i, x in enumerate(itertools.product(X, X)):
            if i % 6 == 0:
                continue
            if x[0] - x[1] == 0:
                X[0] = 1
            else:
                X[0] = 0
        if y==1 or y==2 or y==3 or y==6 or y==7:
            y = 1
        else:
            y = 0
        X[1] = 0
        X[2] = 0
        X[3] = 0
        X[4] = 0
        return {'X': X, 'y': y}
    
# Construindo a Rede Neural com 2 camadas escondidas
class OnePair(torch.nn.Module):
    def __init__(self, I, O):
        super(OnePair, self).__init__()
#        self.norm1 = torch.nn.BatchNorm1d(H1)
#        self.norm2 = torch.nn.BatchNorm1d(H2)
        self.linear1 = torch.nn.Linear(I, O)
#        self.linear2 = torch.nn.Linear(H1, H2)
#        self.linear3 = torch.nn.Linear(H2, O)
#        self.relu = torch.nn.ReLU()
        self.sig = torch.nn.Sigmoid()
    # Feed Foward
    def forward(self, X):
        food = self.linear1(X)
#        food = self.norm1(food)
#        food = self.relu(food)
#        food = self.linear2(food)
#        food = self.norm2(food)
#        food = self.relu(food)
#        food = self.linear3(food)
        food = self.sig(food)
        return food

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = OnePair(5, 1)
#model = model.to(device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)
scheduler = ExponentialLR(optimizer, gamma = 0.8)
n_epochs=50
batch_size=4
stat_report=2500
device = 'cpu'


    
# Treinamento
running_loss = 0.0
running_corrects = 0
since = time.time()
for epoch in range(n_epochs):
    print('Epoch {}/{}'.format(epoch, n_epochs - 1))
    print('-' * 10)
    # Cada época tem sua fase de validação e treinamento
    for phase in ['train', 'test']:
        dataset = PokerDataset(X, 
                               y, 
                               random_state=epoch, 
                               transforms=[ShuffleHand(),
                                          IsPair(),
                                          ToTensor()],
                               mode=phase)
        dataloader = DataLoader(dataset, 
                                batch_size = batch_size, 
                                shuffle = True)
        if phase == 'train':
            scheduler.step()
            model.train()
        if phase == 'test':
            model.eval()

        # Iterando sobre os dados
        for i_batch, sample_batched in enumerate(dataloader):
#                sample_batched['X'] = sample_batched['X'].to(device)
#                sample_batched['y'] = sample_batched['y'].to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(sample_batched['X'])
                loss = criterion(outputs, sample_batched['y'])
                _, prediction = torch.max(outputs, 1)
                prediction = torch.tensor(prediction, dtype = torch.float)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # Estatisticas
                running_loss += loss.item() * sample_batched['X'].size(0)
                running_corrects += torch.sum(
                        prediction == sample_batched['y'])
                # Imprime o custo a cada 10 batches
                if (i_batch % stat_report) == 0:
                    average_loss = running_loss/(batch_size*stat_report)
                    average_corrects = (float(running_corrects)*100)/float(batch_size*stat_report)
                    print('Epoch is {}% Custo: {:.3f} Acc: {}% ({})'.format((i_batch * batch_size)/10000, average_loss, average_corrects, phase))
                    running_loss = 0.0
                    running_corrects = 0
                    
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
time_elapsed // 60, time_elapsed % 60))




list(model.parameters())



dado[1][2] = 1
pair = []
pair_flag = 0
for i, x in enumerate(itertools.product(dado[1].tolist(),dado[1].tolist())):
    if i % 6 == 0:
        continue
    if x[0] - x[1] == 0:
        pair_flag = 1
    print(x)
    
    
        pair_flag = 0
        for i, x in enumerate(itertools.product(X, X)):
            if i % 6 == 0:
                continue
            if x[0] - x[1] == 0:
                pair_flag = 1
len(pair)    
    
    
learn(X, y)



dataset = PokerDataset(X, 
                       y, 
                       random_state=42, 
                       transforms=[ShuffleHand(),
                                   IsPair(),
                                   ToTensor()])
dataloader = DataLoader(dataset, 
                        batch_size = 4, 
                        shuffle = True)
dado = next(iter(dataloader))['X']
print(model(dado))
print(dado)

next(iter(dataloader))
for i, x in enumerate(dataloader):
    print(i, x)

