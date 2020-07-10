import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier1L(nn.Module): #one hidden layer
    
    def __init__(self,embedding_dim,vocab_size,hidden_dim,output_dim):
        
        super(Classifier1L, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc1=nn.Linear(embedding_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,output_dim)
        #Use Cross-Entropy loss requires not calculating probability from the network directly. The output of the last layer is fed to the loss function directly
        
    def forward(self,x): #x is sequence of words (mapped to integers)
        x=x.t()
        lengths = x[0,:] #torch.shape=(batch_size)
        reviews = x[1:,:] #torch.shape=(pad,batch_size)
        out=self.embedding(reviews) #torch.shape=(pad,batch_size,embedding_dim)
        out=torch.cat([torch.sum(out[:l,i,:],0).reshape(1,-1) for i,l in enumerate(lengths)],0)  
        out=F.relu(self.fc1(out))
        out=self.fc2(out)

        return out