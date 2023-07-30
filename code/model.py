# codeing: utf-8
from collections import OrderedDict
from typing import List

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
                 filter_widths: List[int] = [3, 4],
                 num_filters=100,
                 num_conv_layers=3,
                 intermediate_pool_size=3, **kwargs):
        super(CNNTextClassifier, self).__init__()
        self.embedding = nn.Embedding(emb_input_dim, emb_output_dim)
        self.feature_map_count = len(filter_widths) * num_filters

        # Construct convolution layer for each filter width
        self.conv_layers = []
        for width in filter_widths:

            # first convolution layer to be build on
            conv_layers = [
                ('conv0', nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(width, emb_output_dim))),
                ('pooling0', nn.MaxPool2d((intermediate_pool_size, 1)))]

            for i in range(1, num_conv_layers):
                conv_layers = conv_layers + [
                    ('conv' + str(i), nn.Conv2d(in_channels=num_filters, out_channels=num_filters,
                                                kernel_size=(width, 1))),
                    ('pooling' + str(i), nn.MaxPool2d((intermediate_pool_size, 1)))]

            # Global pooling layer
            conv_layers.append(('global_pooling', nn.AdaptiveAvgPool2d(1)))

            # Combine layers using  Sequential
            conv_layers = nn.Sequential(OrderedDict(conv_layers))

            # Store convolution layers for each width
            self.conv_layers.append(conv_layers)

        # To move list of layers to GPU, nn.ModuleList is required
        self.conv_layers = nn.ModuleList(self.conv_layers)

        # Final projection layer
        self.output_layer = nn.Linear(self.feature_map_count, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.xavier_uniform_(module.weight.data)

    def from_embedding(self, embedded, mask):
        mask = mask.unsqueeze(-1)
        # print((mask == 0)[0])
        embedded = embedded.masked_fill(mask == 0, value=0)

        # Add in channel number for convolution layer input
        embedded = embedded.unsqueeze(1)

        # Create feature map and pooling layer.
        # Then concat the outputs of global pooling layer together as input for final dense layer
        pool_output = []
        for i, conv_layer in enumerate(self.conv_layers):
            # Squeeze applied to remove all extra dimensions

            embedded = embedded.float()
            pool_output.append(conv_layer(embedded).squeeze())
        pool_output = torch.cat(pool_output, dim=1)

        # Final dense layer
        output = self.output_layer(pool_output)

        return output

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

    def __init__(self, emb_input_dim=0, emb_output_dim=0, num_classes=2, dr=0.2, dense_units=[100, 100]):
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
        average_pooling = F.avg_pool2d(mask_embed, kernel_size=(seq_length, 1), count_include_pad=True).squeeze(1).to(
            torch.float32)
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
        self.embedding = nn.Embedding(emb_input_dim, emb_output_dim)
        self.num_layers = 2
        self.lstm = nn.LSTM(emb_output_dim, hidden_size, num_layers=self.num_layers, batch_first=True, dropout=dr)
        self.classifier = nn.Linear(hidden_size, num_classes)

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
        mask = mask.unsqueeze(-1)
        mask_embed = embedded.masked_fill(mask == 0, value=0)

        outputs, states = self.lstm(mask_embed)
        outputs = outputs.transpose(2, 1)

        max_pooling = F.adaptive_max_pool2d(outputs, (outputs.size()[-2], 1))
        max_pooling = max_pooling.squeeze(-1)

        classifier = self.classifier(max_pooling)

        return classifier

    def forward(self, data, mask):
        embedded = self.embedding(data)
        return self.from_embedding(embedded, mask)


class LLMLSTMTextClassifier(LSTMTextClassifier):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, llm_embedding, mask):
        return self.from_embedding(torch.squeeze(llm_embedding), mask)



