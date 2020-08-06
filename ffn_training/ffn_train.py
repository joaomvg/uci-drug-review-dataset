import argparse
import json
import os
import pickle
import sys
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data

from model_ffn import Classifier1L, csr_to_torch_float, y_torch

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Classifier1L(model_info['vocab_size'],model_info['hidden_dim'], model_info['output_dim'],model_info['drop_rate'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.to(device).eval()

    print("Done loading model.")
    return model

def _get_train_data_loader(batch_size, training_dir):
    print("Get train data loader.")

    file_path_X=os.path.join(training_dir,"x_train.pkl")
    file_path_y=os.path.join(training_dir,"y_train.pkl")
    x_train=pickle.load(open(file_path_X,'rb'))
    y_train=pickle.load(open(file_path_y,'rb'))
    
    y_train = y_torch(y_train)
    x_train = csr_to_torch_float(x_train)

    train_ds = torch.utils.data.TensorDataset(x_train.to_dense(),y_train)
    
    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)


def train(model, train_loader, epochs, criterion, optimizer, device):
    
    for epoch in range(1, epochs + 1):
        model.train() # Make sure that the model is in training mode.

        total_loss = 0

        for batch in train_loader:
            # get data
            batch_x, batch_y = batch

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()

            # get predictions from model
            y_pred = model(batch_x)
        
            # perform backprop
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data.item()

        print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(train_loader)))



if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments when the script
    # is executed. Here we set up an argument parser to easily access the parameters.

    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Model Parameters
    parser.add_argument('--drop_rate', type=float, default=0.3, metavar='N',
                        help='size of the word embeddings (default: 0.3)')
    parser.add_argument('--hidden_dim', type=int, default=100, metavar='N',
                        help='size of the hidden dimension (default: 100)')
    parser.add_argument('--vocab_size', type=int, default=5000, metavar='N',
                        help='size of the vocabulary (default: 5000)')
    parser.add_argument('--output_dim', type=int, default=300, metavar='N',
                        help='size of the vocabulary (default: 300)')
    
    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)
    print("Train data loaded.")
    
    # Build the model.
    model = Classifier1L(args.vocab_size, args.hidden_dim,args.output_dim,args.drop_rate).to(device)

    print("Model loaded with hidden_dim {}, vocab_size {}, output_dim {}, dropout rate {}".format( args.hidden_dim, args.vocab_size, args.output_dim, args.drop_rate))

    # Train the model.
    optimizer = optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()

    train(model, train_loader, args.epochs, loss_fn,optimizer, device)

    # Save the parameters used to construct the model
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'epochs':args.epochs,
            'hidden_dim': args.hidden_dim,
            'vocab_size': args.vocab_size,
            'output_dim': args.output_dim,
            'drop_rate':args.drop_rate
        }
        torch.save(model_info, f)

	# Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)