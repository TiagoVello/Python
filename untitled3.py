# Construindo a Rede Neural com 3 camadas
class PokerRedeNeural(torch.nn.Module):
    def __init__(self, I, H1, H2, H3, O):
        super(PokerRedeNeural, self).__init__()
        self.norm1 = torch.nn.BatchNorm1d(H1)
        self.norm2 = torch.nn.BatchNorm1d(H2)
        self.norm3 = torch.nn.BatchNorm1d(H3)
        self.linear1 = torch.nn.Linear(I, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, H3)
        self.linear4 = torch.nn.Linear(H3, O)
        self.relu = torch.nn.ReLU()
    
    def forward(self, X):
        food = self.linear1(X)
        food = self.norm1(food)
        food = self.relu(food)
        food = self.linear2(food)
        food = self.norm2(food)
        food = self.relu(food)
        food = self.linear3(food)
        food = self.norm3(food)
        food = self.relu(food)
        food = self.linear4(food)
        return food
    
    
# Construindo a Rede Neural com 5 camadas
class PokerRedeNeural(torch.nn.Module):
    def __init__(self, I, H1, H2, H3, H4, H5,  O):
        super(PokerRedeNeural, self).__init__()
        self.norm1 = torch.nn.BatchNorm1d(H1)
        self.norm2 = torch.nn.BatchNorm1d(H2)
        self.norm3 = torch.nn.BatchNorm1d(H3)
        self.norm4 = torch.nn.BatchNorm1d(H4)
        self.norm5 = torch.nn.BatchNorm1d(H5)
        self.linear1 = torch.nn.Linear(I, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, H3)
        self.linear4 = torch.nn.Linear(H3, H4)
        self.linear5 = torch.nn.Linear(H4, H5)
        self.linear6 = torch.nn.Linear(H5, O)
        self.relu = torch.nn.ReLU()
    # Feed Foward
    def forward(self, X):
        food = self.linear1(X)
        food = self.norm1(food)
        food = self.relu(food)
        food = self.linear2(food)
        food = self.norm2(food)
        food = self.relu(food)
        food = self.linear3(food)
        food = self.norm3(food)
        food = self.relu(food)
        food = self.linear4(food)
        food = self.norm4(food)
        food = self.relu(food)
        food = self.linear5(food)
        food = self.norm5(food)
        food = self.relu(food)
        food = self.linear6(food)
        return food
    
    
class IsPair(object):
    def __call__(self, sample):
        X = sample['X']
        if sample['y'] == (1 or 2 or 3 or 6 or 7):
            y = 1
        else:
            y = 0
        return {'X': X, 'y': y}
    
    
    
# Construindo a Rede Neural com 5 camadas
class OnePair(torch.nn.Module):
    def __init__(self, I, H1, H2, O):
        super(OnePair, self).__init__()
        self.norm1 = torch.nn.BatchNorm1d(H1)
        self.norm2 = torch.nn.BatchNorm1d(H2)
        self.linear1 = torch.nn.Linear(I, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, O)
        self.relu = torch.nn.ReLU()
    # Feed Foward
    def forward(self, X):
        food = self.linear1(X)
        food = self.norm1(food)
        food = self.relu(food)
        food = self.linear2(food)
        food = self.norm2(food)
        food = self.relu(food)
        food = self.linear3(food)
        return food