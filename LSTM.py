import torch.nn as nn
import torch

class LSTM(nn.Module):
    def __init__(self, no_layers, vocab_size, embedding_dim, hidden_dim):
        # Constructor 
        super().__init__()
        self.no_layers = no_layers
        self.hidden_dim = hidden_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM layer(s)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=no_layers, batch_first=True)

        # Linear and Sigmoid Layers
        self.linear = nn.Linear(hidden_dim, 3)
        self.sig = nn.Sigmoid()

    def forward(self, input, hidden_state):
        batch_size = input.size(0)

        embedded_data = self.embedding(input)
        out, hidden = self.lstm(embedded_data, hidden_state)
        out = self.linear(out)
        out = self.sig(out)

        out = out[:, -1]
        return out, hidden

    def init_hidden(self, batch_size):
        h0 = torch.zeros((self.no_layers, batch_size,self.hidden_dim))
        c0 = torch.zeros((self.no_layers, batch_size,self.hidden_dim))
        hidden = (h0,c0)
        return hidden
