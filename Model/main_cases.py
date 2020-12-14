#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 15:02:22 2020

@author: hongru
"""


import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
from torch import nn, optim
from tqdm import trange


#%%%%%%%%
'''
Choose Hyperparameters

Output_size: Choose how many future days to project.

input_size: Number of input features.

sequence_length: Number of past days to fit in the model.

num_layers: Number of LSTM layers.

hidden_layer_size: Hidden layer size for LSTM.

'''
num_epochs =50
output_size = 7
input_size =12 #input_size = number of features
sequence_length = 28 
learning_rate = 0.0001
num_layers = 1
hidden_layer_size = 512

#%%%%%%%%

'''
import data
cases: incidence cases per 10,000 people

'''
df = pd.read_pickle('/Users/hongru/Projects/Covid_projection/data/RNN_input_week44.pickle')
df = df.set_index(['FIPS', 'Date'])
df['cases'] = (df['cases']/df['total_pop'])*10000

#%%

'''
Preprocess data
Scale data between [0,1]

'''
scaler = MinMaxScaler(feature_range = (0, 1))
train_features_normalized = scaler.fit_transform(df.iloc[:, 1:])
scaler_cases = MinMaxScaler(feature_range = (0, 1))
train_cases_normalized = scaler_cases.fit_transform(np.asarray(df.iloc[:,0]).reshape(-1, 1))
df.iloc[:, 1:] = train_features_normalized
df['cases'] = train_cases_normalized

def create_inout_sequences(input_data, seq_length, output_size):
    inout_seq = []
    L = len(input_data)
    for i in range(L-seq_length-output_size+1):
        train_seq = input_data[i:i+seq_length]
        train_label = input_data[i:i+seq_length+output_size][seq_length:seq_length+output_size, 0]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

# Get train and test data for all states
'''
Train on most recent 150 days
'''
train_data = []
state_ordered = []
for i in df.index.get_level_values('FIPS').unique():
    df_state = df.iloc[df.index.get_level_values('FIPS') == i]
    
    if len(df_state) <= 150:    
        train_state = create_inout_sequences(df_state.to_numpy(), sequence_length, output_size)
    
        for x in train_state:
            train_data.append(x)
        state_ordered.append(i)
    else:
        df_state = df.iloc[df.index.get_level_values('FIPS') == i][-150:]
    
        train_state = create_inout_sequences(df_state.to_numpy(), sequence_length, output_size)
    
        for x in train_state:
            train_data.append(x)
        state_ordered.append(i)
#%%%%%%%%%
'''
RNN LSTM 

'''
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, output_size):
#     input_size=3, hidden_layer_size=256, output_size=1):
        super().__init__()
#         torch.manual_seed(0)
        
        self.hidden_layer_size = hidden_layer_size
        
#         self.hidden_size = hidden_layer_size
        
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)

        self.linear = nn.Linear(hidden_layer_size, output_size)
    
        #hidden cell size: (hidden_size, batch_size, hidden_layer_size)
        self.hidden_cell = (torch.zeros(self.num_layers,1,self.hidden_layer_size),
                            torch.zeros(self.num_layers,1,self.hidden_layer_size))


    def forward(self, input_seq):

        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        
        #only return the results for last sequence
        lstm_out = lstm_out[:, -1, :]
        predictions = self.linear(lstm_out)
        return predictions
    
    
#%%%%%%%
'''
select loss function and optimizer here:
    
'''
model = LSTM(input_size, hidden_layer_size, num_layers, output_size)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#%%
'''
Training

'''
def fit(model , optimizer, loss_function, train_data):
    print('Training')
    model.train()
    cumulative_loss = 0
    
    for seq, labels in train_data:
        optimizer.zero_grad()
        seq = torch.tensor(seq).reshape(-1, sequence_length, input_size)
        model.hidden_cell = (torch.zeros(num_layers, 1, model.hidden_layer_size),
                        torch.zeros(num_layers, 1, model.hidden_layer_size))
        y_pred = model(seq.float())
        
        y_pred = y_pred.reshape(output_size)
        
        single_loss = loss_function(y_pred, torch.tensor(labels).float())
        single_loss.backward()
        optimizer.step()
        
        cumulative_loss += single_loss.item()
           
    return cumulative_loss

#%%%
torch.manual_seed(0)
total_loss = []
for i in trange(num_epochs):
    
    train_epoch_loss = fit(model, optimizer, loss_function, train_data)
    total_loss.append(train_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    
#%%

torch.save(model.state_dict(), '/Users/hongru/Projects/Covid_projection/models/RNN-LSTM-7-day-projection_week44.pt')
