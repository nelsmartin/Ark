import torch
import string
from io import open

allowed_characters = string.ascii_letters + ".,:'"

n_letters = len(allowed_characters)

def letterToIndex(letter):
    return allowed_characters.find(letter)


def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


filename = 'small.txt'
lines = open(filename).read().strip().split('\t')



