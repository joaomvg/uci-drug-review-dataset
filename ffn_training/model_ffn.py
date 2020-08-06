import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def csr_to_torch_float(csr):
    coo=csr.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape =coo.shape
    tensor=torch.sparse.FloatTensor(i, v, torch.Size(shape))
    
    return tensor

def y_torch(train_y):
    classes=list(set(train_y))
    classes=sorted(classes)
    dic={target:i for i,target in enumerate(classes)}

    y=np.array([dic[target] for target in train_y])

    return torch.from_numpy(y).squeeze()

class Classifier1L(nn.Module): #one hidden layer
    
    def __init__(self,vocab_size,hidden_dim,output_dim,drop_rate):
        
        super(Classifier1L, self).__init__()
        
        self.fc1=nn.Linear(vocab_size,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,output_dim)
        self.drop=nn.Dropout(drop_rate)
    
        #Use Cross-Entropy loss requires not calculating probability from the network directly. The output of the last layer is fed to the loss function directly
        
    def forward(self,x): #x is sequence of words (mapped to integers)
        out=self.fc1(x)   
        out=F.relu(out)
        out=self.drop(out)
        out=self.fc2(out)

        return out

class Classifier2L(nn.Module): #one hidden layer
    
    def __init__(self,vocab_size,hidden1_dim,hidden2_dim,output_dim,drop_rate=0.3):
        
        super(Classifier2L, self).__init__()
        
        self.fc1=nn.Linear(vocab_size,hidden1_dim)
        self.fc2=nn.Linear(hidden1_dim,hidden2_dim)
        self.fc3=nn.Linear(hidden2_dim,output_dim)
        self.drop=nn.Dropout(drop_rate)
        #Use Cross-Entropy loss requires not calculating probability from the network directly. The output of the last layer is fed to the loss function directly
        
    def forward(self,x): 
        out=self.fc1(x)   
        out=F.relu(out)
        out=self.drop(out)
        out=self.fc2(out)
        out=F.relu(out)
        out=self.drop(out)
        out=self.fc3(out)

        return out