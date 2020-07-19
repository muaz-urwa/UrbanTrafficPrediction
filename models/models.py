import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, feat_size=1, network_size=1, hidden_layer_size=100, lstm_layers=1, dropout=0):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm_layers = lstm_layers
        
        lstm_input = network_size + feat_size
        self.lstm = nn.LSTM(input_size=lstm_input, hidden_size=hidden_layer_size,
                            num_layers=lstm_layers, dropout=dropout)

        self.linear = nn.Linear(hidden_layer_size, network_size)

        
    def initialize_hidden_cell(self, device):
        self.hidden_cell = (torch.zeros(self.lstm_layers,1,self.hidden_layer_size, device=device),
                    torch.zeros(self.lstm_layers,1,self.hidden_layer_size, device=device))

    def forward(self,  input_seq, feat):
        x = torch.cat((input_seq,feat),axis=1)
        lstm_out, self.hidden_cell = self.lstm(x.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions

class LSTM_Pipeline(nn.Module):
    """

    Pipelined Model which performs aggregation, temporal modelling
        and disaggregation. All three steps can be jointly optimized.

    feat_size: dimensions of external features space
    hidden_layer_size: dimensions of slstm output latent space
    network_size: output network dimensions/nodes
    lstm_layers: number of lstm layers
    aggregation_size: dimensions of aggregated space
    at_mat: attachment_matrix initialization
    dropout: dropout value for lstm layer
    tune_at: (bool) to train the agregation or not
    """
    def __init__(self, feat_size=1, hidden_layer_size=100, network_size=1, lstm_layers=1, aggregation_size=10, dropout=0, at_mat=None, tune_at=False):
        super().__init__()
        
        # aggregation
        if at_mat != None:
            self.attachment_matrix = torch.nn.Parameter(at_mat)
            self.attachment_matrix.requires_grad = tune_at
        else:
            self.attachment_matrix = torch.nn.Parameter(torch.randn(network_size,aggregation_size))
            self.attachment_matrix.requires_grad = True
        
        
        self.hidden_layer_size = hidden_layer_size
        self.lstm_layers = lstm_layers
        # self.initialize_hidden_cell()
        
        
        lstm_input = aggregation_size + feat_size
        self.lstm = nn.LSTM(input_size=lstm_input, hidden_size=hidden_layer_size, num_layers=lstm_layers, dropout=dropout)

        #disaggregation
        self.linear_2 = nn.Linear(hidden_layer_size, network_size)

    def initialize_hidden_cell(self, device):
        self.hidden_cell = (torch.zeros(self.lstm_layers,1,self.hidden_layer_size, device=device),
                    torch.zeros(self.lstm_layers,1,self.hidden_layer_size, device=device))

    def forward(self, input_seq, feat):
        
        # aggregation
        w = F.softmax(self.attachment_matrix, dim=1)
        x = torch.matmul(input_seq, self.attachment_matrix)
        x = torch.cat((x,feat),axis=1)

        #temporal modelling
        lstm_out, self.hidden_cell = self.lstm(x.view(len(input_seq) ,1, -1), self.hidden_cell)
        
        # disaggregation
        predictions = self.linear_2(lstm_out.view(len(input_seq), -1))
        
        return predictions