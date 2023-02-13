# codeing: utf-8

import argparse
import logging
import numpy as np
import torch
import sys
from sklearn.metrics import precision_recall_curve, average_precision_score

from load_data import get_data_loaders, get_hf_data_loaders
from model import CNNTextClassifier, LSTMTextClassifier, DANTextClassifier
from model import LLMCNNTextClassifier, LLMLSTMTextClassifier, LLMDANTextClassifier
from utils import logging_config, get_device
from torchtext.vocab import pretrained_aliases


parser = argparse.ArgumentParser(description='Train a (short) text classifier - via convolutional or other standard architecture')
parser.add_argument('--train_file', type=str, help='File containing file representing the input TRAINING data')
parser.add_argument('--val_file', type=str, help='File containing file representing the input VALIDATION data', default=None)
parser.add_argument('--test_file', type=str, help='File containing file representing the input TEST data', default=None)
parser.add_argument('--epochs', type=int, default=10, help='Upper epoch limit')
parser.add_argument('--lr',type=float, help='Learning rate', default=0.001)
parser.add_argument('--batch_size',type=int, help='Training batch size', default=32)
parser.add_argument('--dropout', type=float, help='Dropout ratio', default=0.5)
parser.add_argument('--seq_length', type=int, help='Max sequence length', default = 128)
parser.add_argument('--embedding_source', type=str, default='6B', help='Pre-trained embedding source name')
parser.add_argument('--lstm', action='store_true', help='Use an LSTM layer instead of CNN encoder')
parser.add_argument('--dan', action='store_true', help='Use a DAN enocder instead of CNN encoder')
parser.add_argument('--dense_dan_layers', type=str, default='100,100', help='List if integer dense unit layers (DAN only)')
parser.add_argument('--log_dir', type=str, default='.', help='Output directory for log file')
parser.add_argument('--fixed_embedding', action='store_true', help='Fix the embedding layer weights')
parser.add_argument('--random_embedding', action='store_true', help='Use random initialized embedding layer')
parser.add_argument('--embedding_size', type=int, default=200, help='Embedding size (if random) (DEFAULT = 200)')
parser.add_argument('--filter_sizes', type=str, default='3,4', help='List of integer filter sizes (for CNN only)')
parser.add_argument('--num_filters', type=int, default=100, help='Number of filters (of each size)')
parser.add_argument('--num_conv_layers', type=int, default=3, help='Number of convolutional/pool layers')
parser.add_argument('--use_llm_embeddings', action='store_true', help='Use an LLM as a feature extractor (no fine-tuning)', default=False)
parser.add_argument('--clip', type=float, default=10.0, help='Gradient clipping value')
parser.add_argument('--llm_cache_dir', type=str, help='Directory to store bert embeddings', default=None)
parser.add_argument('--llm_proj_dim', type=int, help='Down-projected dimensions (0 indicates no projection is performed)', default=0)

args = parser.parse_args()
loss_fn = torch.nn.CrossEntropyLoss()


def get_llm_based_model(num_classes, embedding_size):
    if args.lstm:
        model = LLMLSTMTextClassifier(emb_output_dim=embedding_size, num_classes=num_classes, dr=args.dropout)
    elif args.dan:
        model = LLMDANTextClassifier(emb_output_dim=embedding_size, num_classes=num_classes, dr=args.dropout)
    else:
        model = LLMCNNTextClassifier(emb_output_dim=embedding_size, filter_widths=filters,
                                          num_conv_layers=args.num_conv_layers,
                                          num_filters=args.num_filters,
                                          num_classes=num_classes, dr=args.dropout)
    return model


def get_model(num_classes, vocab_size=0, embedding_size=0, pretrained_vectors=None):
    filters = [ int(x) for x in args.filter_sizes.split(',') ]
    emb_input_dim, emb_output_dim = vocab_size, embedding_size
    if args.lstm:
        model = LSTMTextClassifier(emb_input_dim=emb_input_dim, emb_output_dim=emb_output_dim,
                                       num_classes=num_classes, dr=args.dropout)
    elif args.dan:
        dense_units = [ int(x) for x in args.dense_dan_layers.split(',') ]
        model = DANTextClassifier(emb_input_dim=emb_input_dim,
                                  emb_output_dim=emb_output_dim,
                                  num_classes=num_classes, dr=args.dropout, dense_units=dense_units)
    else:
        model = CNNTextClassifier(emb_input_dim=emb_input_dim, emb_output_dim=emb_output_dim,
                                      filter_widths=filters, num_classes=num_classes,
                                      dr=args.dropout, num_conv_layers=args.num_conv_layers,
                                      num_filters=args.num_filters)
    if pretrained_vectors is not None:
        model.embedding.weight.data.copy_(pretrained_vectors)
        if args.fixed_embedding:
            model.embedding.weight.requires_grad = False
    return model


def train_classifier(model, train_loader, val_loader, test_loader, num_classes, device):
    trainer = torch.optim.Adam(model.parameters(), lr = args.lr)
    for epoch in range(args.epochs):
        epoch_loss = 0
        for i, (label, data, mask) in enumerate(train_loader):
            # note - data will be a tensor of token IDs if the model is a non-LLM-based model
            #      - data will be a tensor of embedding sequences if the model is LLM-based
            data = data.to(device)
            label = label.to(device)
            output = model(data, mask)
            l = loss_fn(output, label).mean()
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip) # sometimes helpful to clip gradients
            trainer.step()
            trainer.zero_grad()
            epoch_loss += l.item()
        logging.info("Epoch {} loss = {}".format(epoch+1, epoch_loss))
        tr_acc = evaluate(model, train_loader, device)
        logging.info("TRAINING Acc = {}".format(tr_acc))        
        val_acc = evaluate(model, val_loader, device)
        logging.info("VALIDATION Acc = {}".format(val_acc))
    if test_loader is not None:
        tst_acc = evaluate(model, test_loader, device)
        logging.info("***** Training complete. *****")
        logging.info("TEST Acc = {}".format(tst_acc))
        

def evaluate(model, dataloader, device):
    """
    Get predictions on the dataloader items from model
    Return metrics (accuracy, etc.)
    """
    acc = 0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for i, (label, ids, mask) in enumerate(dataloader):
            out = model(ids, mask)
            out_argmax = out.argmax(1)
            total_correct += (out_argmax == label).sum().item()
            total += label.size(0)
            acc = total_correct / float(total)
    return 0.0, acc
    

if __name__ == '__main__':
    logging_config(args.log_dir, 'train', level=logging.INFO)
    label_map = {"Objectives": 0, "Outcome": 1, "Prior": 2, "Approach": 3}

    if args.use_llm_embeddings:
        train_loader, val_loader, test_loader = \
            get_hf_data_loaders(args.train_file,
                                args.val_file,
                                args.test_file,
                                args.batch_size,
                                args.seq_length,
                                label_map,
                                cache_dir=args.llm_cache_dir,
                                proj_size=args.llm_proj_dim)
        embedding_size = args.llm_proj_dim or 768
        model = get_llm_based_model(len(label_map), embedding_size)
    else:
        vocab, train_loader, val_loader, test_loader = \
            get_data_loaders(args.train_file,
                             args.val_file,
                             args.test_file,
                             args.batch_size,
                             args.seq_length,
                             label_map)
        pretrained, embedding_size = None, args.embedding_size
        if not args.random_embedding:
            embedding = pretrained_aliases[args.embedding_source]()
            pretrained = embedding.get_vecs_by_tokens(vocab.get_itos())
            embedding_size = pretrained.size()[-1]
        vocab_size = len(vocab)
        model = get_model(len(label_map), vocab_size=vocab_size, embedding_size=embedding_size, pretrained_vectors=pretrained)
    device = get_device()
    model.to(device)
    train_classifier(model, train_loader, val_loader, test_loader, len(label_map), device=device)
