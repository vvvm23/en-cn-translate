import torch
import torch.nn
import torch.nn.functional as F

from dataloader import LangDataset
from model import LangTransformer

import tqdm
import time

# some constant parameters
TRY_CUDA = True
NB_EPOCHS = 100
BATCH_SIZE = 64
NB_TEST = 1000
NB_HIDDEN = 2056
EMD_DIM = 1024 # should be a multiple of 8 (or nb_heads)

# Get the CUDA device if available
device = torch.device('cuda:0' if TRY_CUDA and torch.cuda.is_available() else 'cpu')
print(f"> Device: {device} ({'CUDA is enabled' if TRY_CUDA and torch.cuda.is_available() else 'CUDA not available'}) \n")

# Initialise the language dataset
dataset = LangDataset("data/cmn.txt")
src_dim = dataset.token_in.nb_words
tgt_dim = dataset.token_out.nb_words

# Randomly split the dataset into a test and train set
test_dataset, train_dataset = torch.utils.data.random_split(dataset, [NB_TEST, len(dataset) - NB_TEST])
print(f"> Train set size: {len(train_dataset)}")
print(f"> Test set size: {len(test_dataset)}\n")

# Initialise the dataloaders for the respective partitions
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

# Intialise the Transformer model
model = LangTransformer(src_dim, tgt_dim, EMD_DIM, NB_HIDDEN, device).to(device)
model.train()

# Define criteria and optimiser
crit = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.0001)

# Function that evaluates the model and returns the average loss
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0.0

    for i, (src, tgt, out) in enumerate(dataloader):
        # Move data to target device
        src = src.type(torch.LongTensor).to(device)
        tgt = tgt.type(torch.LongTensor).to(device)
        out = out.type(torch.LongTensor).to(device)

        pred = model(src, tgt)

        # smush batch and time dimensions into one single dimension    
        pred = pred.reshape(-1, tgt_dim)
        out = out.reshape(-1)

        # ignore the last of output and ignore the SOS in tgt (my version of shifting to the right)
        loss = crit(pred, out)
        total_loss += loss.item()

    model.train()
    return total_loss / len(dataloader)

save_id = int(time.time())

for ei in range(NB_EPOCHS):
    print(f"> Epoch {ei+1}/{NB_EPOCHS}")
    total_loss = 0.0
    pbar = tqdm.tqdm(total=len(train_dataloader))
    for i, (src, tgt, out) in enumerate(train_dataloader):
        # Move data to target device
        src = src.type(torch.LongTensor).to(device)
        tgt = tgt.type(torch.LongTensor).to(device)
        out = out.type(torch.LongTensor).to(device)

        # Zero the optimiser
        optim.zero_grad()

        pred = model(src, tgt)
    
        # smush batch and time dimensions into one single dimension    
        pred = pred.reshape(-1, tgt_dim)
        out = out.reshape(-1)

        # ignore the last of output and ignore the SOS in tgt (my version of shifting to the right)
        loss = crit(pred, out)

        # increment total training loss and backpropagate
        total_loss += loss.item()
        loss.backward()

        # Step the optimiser and progress bar
        optim.step()
        pbar.update(1)
    
    print(f"> Training Loss: {total_loss / len(train_dataloader)}")
    print(f"> Validation Loss: {evaluate(model, test_dataloader)}\n")

    # Save the model at the end of every epoch
    torch.save(model, f"models/{save_id}-{ei}.pt")

    total_loss = 0.0

print("> Training complete.")
