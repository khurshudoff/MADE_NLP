import torch
import torch.nn as nn
import torch.optim as optim

from torchnlp import nn as nlpnn

import torchtext
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import random
import math
import time


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=emb_dim
        )
        
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, src):
        
        #src = [src sent len, batch size]
        
        # Compute an embedding from the src data and apply dropout to it
        embedded = self.embedding(src)# <YOUR CODE HERE>
        
        embedded = self.dropout(embedded)
        
        output, hidden = self.lstm(embedded)
        #embedded = [src sent len, batch size, emb dim]
        
        # Compute the RNN output values of the encoder RNN. 
        # outputs, hidden and cell should be initialized here. Refer to nn.LSTM docs ;)
        
        # <YOUR CODE HERE> 
        
        #outputs = [src sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        
        output = (output[:, :, :self.hid_dim] +
                  output[:, :, self.hid_dim:])
        
        return output, hidden
    
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(
            num_embeddings=output_dim,
            embedding_dim=emb_dim
        )
            # <YOUR CODE HERE>
        
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout,
        )
            # <YOUR CODE HERE>
        
        self.attn = nlpnn.Attention(hid_dim)
        
        self.out = nn.Linear(
            in_features=2 * hid_dim,
            out_features=output_dim
        )
            # <YOUR CODE HERE>
        
        self.dropout = nn.Dropout(p=dropout)# <YOUR CODE HERE>
        
    def forward(self, input, hidden, encoder_output):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        # Compute an embedding from the input data and apply dropout to it
        embedded = self.dropout(self.embedding(input))# <YOUR CODE HERE>
        
        #embedded = [1, batch size, emb dim]
        
        # Compute the RNN output values of the encoder RNN. 
        # outputs, hidden and cell should be initialized here. Refer to nn.LSTM docs ;)
        # <YOUR CODE HERE>
        
        
        #output = [sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #sent len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        
        output, hidden = self.lstm(embedded, hidden)
        
        attn_output, _ = self.attn(output.transpose(0, 1),
                              encoder_output.transpose(0, 1))
        
        attn_output = attn_output.transpose(0, 1)
        
        prediction = self.out(torch.cat([attn_output.squeeze(0),
                                         output.squeeze(0)], dim=1))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        self.linear_lt = nn.Linear(
          in_features=self.encoder.hid_dim * 2,
          out_features=self.encoder.hid_dim,
        )
        
        self.linear_st = nn.Linear(
          in_features=self.encoder.hid_dim * 2,
          out_features=self.encoder.hid_dim,
        )
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src sent len, batch size]
        #trg = [trg sent len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        # Again, now batch is the first dimention instead of zero
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_output, hidden = self.encoder(src)
        
        h1 = hidden[0].view(self.encoder.n_layers, src.shape[1], self.encoder.hid_dim * 2)
        h2 = hidden[1].view(self.encoder.n_layers, src.shape[1], self.encoder.hid_dim * 2)
        
        h1 = self.linear_lt(h1).contiguous()
        h2 = self.linear_st(h2).contiguous()
    
        hidden = (h1, h2)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, max_len):
            
            output, hidden = self.decoder(input, hidden, encoder_output)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)
        
        return outputs
    
    
#         max_len = trg.shape[0]
#         trg_vocab_size = dec.output_dim

#         #tensor to store decoder outputs
#         outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(device)

#         #last hidden state of the encoder is used as the initial hidden state of the decoder
#         encoder_output, hidden = enc(src)

#         hidden = hidden[:2]

#         #first input to the decoder is the <sos> tokens
#         input = trg[0,:]

#         for t in range(1, max_len):

#             output, hidden = dec(input, hidden, encoder_output)
#             outputs[t] = output
#             teacher_force = random.random() < teacher_forcing_ratio
#             top1 = output.max(1)[1]
#             input = (trg[t] if teacher_force else top1)