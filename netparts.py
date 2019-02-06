import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import scipy.io
from torch.utils.data import Dataset
import pickle
import os
import sys
# Data Processing

def load_file(path_to_file):
    return np.genfromtxt(path_to_file,delimiter='\t',skip_header=4).astype('float32')



class WalkDataset(Dataset):
    """Walking dataset"""

    def __init__(self, root_dir, sequence_length = 30, transform=None):
        """
        Args:
            root_dir (string): Path to the database folder.
            transform (callable, optional): Optional transform to be applied on
            a sample.
        """
        files = os.listdir(root_dir)
        for i, f in enumerate(files):
            files[i]=os.path.join(root_dir,f)
        self.files = files
        self.transform = transform
        f = open('stats','rb')
        self.mu, self.sigma = pickle.load(f)
        f.close()
        self.sl = sequence_length
        self.len = None # Total number of fixed length sequences
        self.file_len = [0]*len(files)  # Number of fixed length sequences in each file
        self.len_cum = [0]*(len(files)+1) # Number of acumulated sequences

    def load_file(path_to_file):
        return np.genfromtxt(path_to_file,delimiter='\t',skip_header=4).astype('float32')

    def __len__(self):
        if self.len is not None:
            return self.len
        else:
            # Calculate length of the entire fixed length dataset
            for i, name in enumerate(self.files):
                temp = load_file(name)
                sl = temp.shape[0] # Number of timesteps
                self.file_len[i] = sl//(self.sl+1) # Number of fixed length sequences in the file
                self.len_cum[i+1] = np.sum(self.file_len)
            self.len = np.sum(self.file_len)
            return self.len


    def __getitem__(self, idx):
        data = []
        target = []
        #data_lengths = []
        idxs = np.arange(len(self))
        idxs = idxs.tolist()

        if isinstance(idx, slice):
            idxs = idxs[idx]
        else:
            idxs = [idxs[idx]]

        last_file = -1

        for i, n in enumerate(idxs):
            if i>=self.len:
                raise IndexError('The requested sequence does not exist')
            top = self.len_cum[1]
            file_n = 0
            while top-1 < n:
                file_n += 1
                top = self.len_cum[file_n+1]
            if last_file != file_n:
                t = load_file(self.files[file_n])
                t = np.delete(t, np.s_[-3:],1) # Delete the last 3 columns
                #t = np.delete(t, np.s_[self.file_len[file_n]*(self.sl+1):],0) # Delete extra timesteps
                t = np.divide((t-self.mu), self.sigma) # Normalize data
                out_t = np.delete(t, np.s_[:18],1) # Delete Rigth Leg Data
                last_file = file_n
            actual = n + 1 - self.len_cum[file_n]
            input_t = t[(actual-1)*self.sl:actual*self.sl,:]
            output_t = out_t[(actual-1)*self.sl+1:actual*self.sl+1,:]
            #print('first file: '+self.files[0])
            #print ('file name: '+self.files[file_n])
            #print('data size: {}, target size {}'.format(input_t.shape, output_t.shape))
            #sys.stdout.flush()
            data.append(input_t)
            target.append(output_t)
        if len(data)>1:
            data = np.stack(data, axis = 1) # Batch Dimension
            target = np.stack(target, axis = 1)
        else:
            data = data[0]
            target = target[0]
        data = torch.from_numpy(data)
        target = torch.from_numpy(target)
        #data = Variable(data, requires_grad = False)
        #target = Variable(target, requires_grad = False)
        sample = {'data':data, 'target':target}

        return sample



#        for i in range(len(list_files)):
#            t = load_file(list_files[i])
#            t = np.delete(t,np.s_[-3:],1) # Delete the last 3 columns
#            input_t = np.delete(t,np.s_[-1],0) # Delete last element
#            input_t = np.divide((input_t-self.mu),self.sigma) # Normalize data
#            output_t = np.delete(t,np.s_[0],0) # Delete first element
#            output_t = np.divide((output_t -self.mu),self.sigma) # Normalize data
#            output_t = np.delete(output_t,np.s_[:18],1) # Delete Right Leg data
#            data.append(input_t)
#            data_lengths.append(input_t.shape[0]) # Sequence length
#            target.append(output_t)
#
#        largest = max(data_lengths)
#        container = torch.zeros((len(data),largest,36))
#        target_container = torch.zeros((len(data),largest,18))
#        for i in range(len(data)):
#            input_t = data[i]
#            output_t = target[i]
#            extra = largest-input_t.shape[0]
#            container[i] = torch.from_numpy(np.concatenate([input_t,np.zeros((extra,input_t.shape[1]),dtype=input_t.dtype)],0))
#            target_container[i] = torch.from_numpy(np.concatenate([output_t,np.zeros((extra,output_t.shape[1]),dtype=output_t.dtype)],0))
#        container = Variable(container, requires_grad = False)
#        target_container = Variable(target_container, requires_grad = False) 
#        data_packed = nn.utils.rnn.pack_padded_sequence(container, data_lengths,
#                                                        batch_first=True)
#        target_packed = nn.utils.rnn.pack_padded_sequence(target_container, data_lengths,
#                                                        batch_first=True)
#
#        sample = {'data':data_packed, 'target':target_packed}
#
#        return sample

# Main model

class Net(nn.Module):
    def __init__(self, hidden_dim):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(36, hidden_dim, 1, batch_first = True, dropout = 0.5)
        self.fc1 = nn.Linear(hidden_dim, 18)
        self.fc2 = nn.Linear(100,18)
        self.dp = nn.Dropout()

    def forward(self, x, hc):
        #print('input:{}, h1: {}, h2: {}'.format(x.size(),hc[0].size(),hc[1].size()))
        #sys.stdout.flush()
        o, hc = self.lstm(x, hc)
        #o_unpacked, o_unpacked_length = nn.utils.rnn.pad_packed_sequence(o, batch_first = True)
        #x_unpacked, x_unpacked_length = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)
        #x_l = torch.chunk(x_unpacked, 2, dim = 2)
        x_l = torch.chunk(x, 2, dim = 2)
        x_o = x_l[1] # Left Leg data
        #o = F.relu(self.fc1(o_unpacked))
        #o = F.relu(self.fc1(o))
        o = self.fc1(o)
        #o = self.dp(o)
        #o = self.fc2(o)
        o = x_o + o
        #print(o.size())
        #sys.stdout.flush()
        #o = nn.utils.rnn.pack_padded_sequence(o, o_unpacked_length, batch_first=True) 
        return o, hc

    def init_hidden(self,x):
        #batch_size = x.batch_sizes
        #batch_size = batch_size[0]
        batch_size = x.size()[0]
        h_0 = torch.zeros(1, batch_size, self.hidden_dim)
        c_0 = torch.zeros(1, batch_size, self.hidden_dim)
        return (h_0, c_0)
