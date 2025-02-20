import torch
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import string
import time


class NamesDataset(Dataset):

    def __init__(self, filename):
        self.load_time = time.localtime
        labels_set = set()
        self.filename = filename

        self.data = []
        self.data_tensors = []
        self.labels = []
        self.labels_tensors = []

        lines = open(filename).read().strip().split('\t')
        



