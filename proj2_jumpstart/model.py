# codeing: utf-8
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class CNNTextClassifier(nn.Module):
    """
    Parameters
    ----------
    emb_input_dim : int
        The dimensionality of the vocabulary (input to Embedding layer)
    emb_output_dim : int
        The dimensionality of the embedding
    num_classes : int, default 2
        Number of categories in classifier output    
    dr : float, default 0.2
        Dropout rate
    filter_widths : list of int, default = [3,4]
        The widths for each set of filters
    num_filters : int, default = 100
        Number of filters for each width
    num_conv_layers : int, default = 3
        Number of convolutional layers (conv + pool)
    intermediate_pool_size: int, default = 3
    """
    def __init__(self, emb_input_dim=0, emb_output_dim=0,
                 num_classes=2,
                 dr=0.2,
                 filter_widths=[3,4],
                 num_filters=100,
                 num_conv_layers=3,
                 intermediate_pool_size=3, **kwargs):
        super(CNNTextClassifier, self).__init__(**kwargs)
        ## .. TODO .. 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.xavier_uniform_(module.weight.data)

    def from_embedding(self, embedded, mask):
        ## .. TODO ..
        pass

    def forward(self, data, mask):
        embedded = self.embedding(data)
        return self.from_embedding(embedded, mask)


class LLMCNNTextClassifier(CNNTextClassifier):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, llm_embedding, mask):
        return self.from_embedding(torch.squeeze(llm_embedding), mask)


class DANTextClassifier(nn.Module):
    """
    Parameters
    ----------
    emb_input_dim : int
        The dimensionality of the vocabulary (input to Embedding layer)
    emb_output_dim : int
        The dimensionality of the embedding
    num_classes : int, default 2
        Number of categories in classifier output    
    dr : float, default 0.2
        Dropout rate
    dense_units : list[int], default = [100,100]
        Dense units for each layer after pooled embedding
    """
    def __init__(self, emb_input_dim=0, emb_output_dim=0, num_classes=2, dr=0.2, dense_units=[100,100]):
        super(DANTextClassifier, self).__init__()
        self.embedding = nn.Embedding(emb_input_dim, emb_output_dim)

        # Construct multiple feedforward blocks based on the dense units

        feedforward_layers = [('dropout0', nn.Dropout(dr)), ('line0', nn.Linear(emb_output_dim, dense_units[0])),
                              ('relu0', nn.ReLU())]
        for i in range(1, len(dense_units)):
            feedforward_layers = feedforward_layers + [('dropout' + str(i), nn.Dropout(dr)),
                                                       ('line' + str(i), nn.Linear(dense_units[i - 1], dense_units[i])),
                                                       ('relu' + str(i), nn.ReLU())]
        self.feedforward_layers = nn.Sequential(OrderedDict(feedforward_layers)).float()

        # final projection layer
        self.classifier = nn.Linear(dense_units[-1], num_classes)

        self.apply(self._init_weights)
        self.apply(self._init_weights)
        
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight.data)            
            module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight.data)                        

    def from_embedding(self, embedding, mask):
        mask = mask.unsqueeze(-1)

        mask_embed = embedding.masked_fill(mask == 0, value=0)
        seq_length = mask_embed.size()[1]
        average_pooling = F.avg_pool2d(mask_embed, kernel_size=(seq_length, 1), count_include_pad=True).squeeze(1).to(torch.float32)
        linear = self.feedforward_layers(average_pooling)

        classifier = self.classifier(linear)

        return classifier
    
    def forward(self, data, mask):
        embedded = self.embedding(data)
        return self.from_embedding(embedded, mask)


class LLMDANTextClassifier(DANTextClassifier):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, llm_embedding, mask):
        return self.from_embedding(torch.squeeze(llm_embedding), mask)
    


class LSTMTextClassifier(nn.Module):
    """
    Parameters
    ----------
    emb_input_dim : int
        The dimensionality of the vocabulary (input to Embedding layer)
    emb_output_dim : int
        The dimensionality of the embedding
    hidden_size : int
        Dimension size for hidden states within the LSTM
    num_classes : int, default 2
        Number of categories in classifier output    
    dr : float, default 0.2
        Dropout rate
    """
    def __init__(self, emb_input_dim=0, emb_output_dim=0, hidden_size=100, num_classes=2, dr=0.2):
        super(LSTMTextClassifier, self).__init__()

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight.data)            
            module.bias.data.zero_()
        elif isinstance(module, nn.LSTM):
            for i in range(self.num_layers):
                torch.nn.init.xavier_uniform_(module.all_weights[i][0])
                torch.nn.init.xavier_uniform_(module.all_weights[i][1])
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight.data)                        
        

    def from_embedding(self, embedded, mask):
        pass

    def forward(self, data, mask):
        embedded = self.embedding(data)
        return self.from_embedding(embedded, mask)


class LLMLSTMTextClassifier(LSTMTextClassifier):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, llm_embedding, mask):
        return self.from_embedding(torch.squeeze(llm_embedding), mask)
    
    
    
