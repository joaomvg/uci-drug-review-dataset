import torch
import torch.nn as nn
import numpy as np

def csr_to_torch_long(csr):
    coo=csr.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.LongTensor(values)
    shape =coo.shape
    tensor=torch.sparse.LongTensor(i, v, torch.Size(shape))
    
    return tensor

def y_torch(train_y):
    classes=list(set(train_y))
    classes=sorted(classes)
    dic={target:i for i,target in enumerate(classes)}

    y=np.array([dic[target] for target in train_y])

    return torch.from_numpy(y).squeeze()

class LSTMClassifier(nn.Module):
    """
    This is the simple RNN model we will be using to perform Sentiment Analysis.
    """

    def __init__(self, vocab_size,embedding_dim, hidden_dim,output_dim):
        """
        Initialize the model by settingg up the various layers.
        """
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.dense = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        
    def forward(self, x):
        """
        Perform a forward pass of our model on some input.
        """
        x = x.t()
        lengths = x[0,:]
        reviews = x[1:,:]
        embeds = self.embedding(reviews)
        lstm_out, _ = self.lstm(embeds)
        out = self.dense(lstm_out)
        out = out[lengths - 1, range(len(lengths))]
        return out