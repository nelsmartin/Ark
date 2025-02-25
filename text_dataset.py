import torch
import string
from io import open
import unicodedata
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import string
import time
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in allowed_characters
    )
char_set = set()
with open('small.txt', 'r') as file:
            for line in file:
                for i in range(len(line)):
                    char_set.add(line[i])
extras = string.ascii_letters + ".,:'0123456789"
chars = ""

for i in range(len(extras)):
    char_set.add(extras[i])
    
for letter in char_set:
    chars += letter

                    
allowed_characters = chars

n_letters = len(allowed_characters)
def letterToIndex(letter):
    return allowed_characters.find(letter)


def lineToTensor(line):
    line = unicodeToAscii(line)
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

# Check if CUDA is available
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

torch.set_default_device(device)


class TextDataset(Dataset):

    def __init__(self, filename):
        self.load_time = time.localtime
        labels_set = set()
        self.filename = filename

        self.data = []
        self.data_tensors = []
        self.labels = []
        self.labels_tensors = []


        with open(filename, 'r') as file:
            for line in file:
                line = line.strip().split('\t')
                
                self.data.append(line[0])
                self.data.append(line[1])
                self.data_tensors.append(lineToTensor(line[0]))
                self.data_tensors.append(lineToTensor(line[1]))

                self.labels.append('A')
                self.labels.append('B')

        labels_set.add('A')
        labels_set.add('B')
        self.labels_uniq = list(labels_set)
        # Creates a list of tensor labels, one for each data tensor. 
        for idx in range(len(self.labels)):
            temp_tensor = torch.tensor([self.labels_uniq.index(self.labels[idx])], 
            dtype=torch.long)
            self.labels_tensors.append(temp_tensor)

    def __len__(self):
            return len(self.data)
        
    def __getitem__(self, idx):
        data_item = self.data[idx]
        data_label = self.labels[idx]
        data_tensor = self.data_tensors[idx]
        label_tensor = self.labels_tensors[idx]

        return label_tensor, data_tensor, data_label, data_item
    
filename = 'small.txt'
data_set = TextDataset(filename)

train_set, test_set = torch.utils.data.random_split(data_set, [.85, .15], generator=torch.Generator(device=device).manual_seed(2024))

print(f"train examples = {len(train_set)}, validation examples = {len(test_set)}")



class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, line_tensor):
        rnn_out, hidden = self.rnn(line_tensor)
        output = self.h2o(hidden[0])
        output = self.softmax(output)

        return output
    

n_hidden = 128
rnn = CharRNN(n_letters, n_hidden, len(data_set.labels_uniq))
print(rnn)


def label_from_output(output, output_labels):
    top_n, top_i = output.topk(1)
    label_i = top_i[0].item()
    return output_labels[label_i], label_i

def train(rnn, training_data, n_epoch = 10, n_batch_size = 64, report_every = 50, learning_rate = 0.2, criterion = nn.NLLLoss()):
    """
    Learn on a batch of training_data for a specified number of iterations and reporting thresholds
    """
    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    rnn.train()
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

    start = time.time()
    print(f"training on data set with n = {len(training_data)}")

    for iter in range(1, n_epoch + 1):
        rnn.zero_grad() # clear the gradients

        # create some minibatches
        # we cannot use dataloaders because each of our names is a different length
        batches = list(range(len(training_data)))
        random.shuffle(batches)
        batches = np.array_split(batches, len(batches) //n_batch_size )

        for idx, batch in enumerate(batches):
            batch_loss = 0
            for i in batch: #for each example in this batch
                (label_tensor, text_tensor, label, text) = training_data[i]
                output = rnn.forward(text_tensor)
                loss = criterion(output, label_tensor)
                batch_loss += loss

            # optimize parameters
            batch_loss.backward()
            nn.utils.clip_grad_norm_(rnn.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()

            current_loss += batch_loss.item() / len(batch)

        all_losses.append(current_loss / len(batches) )
        if iter % report_every == 0:
            print(f"{iter} ({iter / n_epoch:.0%}): \t average batch loss = {all_losses[-1]}")
        current_loss = 0

    return all_losses


start = time.time()
all_losses = train(rnn, train_set, n_epoch=27, learning_rate=0.15, report_every=1)
end = time.time()
print(f"training took {end-start}s")


