# coding: utf-8
import io
import orjson
import torch
import os
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from utils import get_device
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import AutoTokenizer, BertModel
from transformers import AutoTokenizer, OpenAIGPTModel
from pathlib import Path
from sklearn.random_projection import SparseRandomProjection
import numpy as np


def get_data(json_l_file, label_key="label0", text_key="sentence"):
    data = []
    with io.open(json_l_file) as fp:
        for line in fp:
            js_obj = orjson.loads(line)
            data.append((js_obj["label0"], js_obj["sentence"]))
    return data


def get_vocab_and_pipeline(train_data):
    tokenizer = get_tokenizer('basic_english')

    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(yield_tokens(iter(train_data)), specials=["<pad>", "<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    text_pipeline = lambda x: vocab(tokenizer(x))
    return vocab, text_pipeline


def get_loader(data, text_pipeline, label_pipeline, batch_size, max_len, shuffle=False):
    device = get_device()

    def collate_batch(batch):
        label_list, text_list, mask_list = [], [], []
        for (_label, _text) in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            tlen = processed_text.size(0)
            if tlen >= max_len:
                processed_text = processed_text[:max_len]
            else:
                processed_text = torch.nn.functional.pad(processed_text, (0, (max_len - tlen)), "constant", 0)
            mask = torch.full(processed_text.shape, 1)
            mask[tlen:] = 0
            mask_list.append(mask)
            text_list.append(processed_text)
        label_list = torch.tensor(label_list, dtype=torch.int64)
        text_list = torch.vstack(text_list)
        mask_list = torch.vstack(mask_list)
        return label_list.to(device), text_list.to(device), mask_list.to(device)

    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch)


def get_data_loaders(train_file, val_file, test_file, batch_size, max_len, label_map):
    train_data = get_data(train_file)
    val_data = get_data(val_file)
    test_data = get_data(test_file)

    label_pipeline = lambda x: label_map.get(x, 0)
    vocab, text_pipeline = get_vocab_and_pipeline(train_data)

    train_dataloader = get_loader(train_data, text_pipeline, label_pipeline, batch_size, max_len, shuffle=True)
    val_dataloader = get_loader(val_data, text_pipeline, label_pipeline, batch_size, max_len, shuffle=False)
    test_dataloader = get_loader(test_data, text_pipeline, label_pipeline, batch_size, max_len, shuffle=False)

    return vocab, train_dataloader, val_dataloader, test_dataloader


########################################################################################################
#### The code ABOVE is for creating dataloaders for use with static word (type) embeddings (e.g. Glove).
####
#### The code BELOW is for dataloaders that assume a Large Language Model (LLM) embedding model required
#### for Assignment 2.
########################################################################################################


def get_hf_loader(data, embedding_model, text_pipeline, label_pipeline, batch_size, max_len,
                  shuffle=False, projection=None):
    """ Construct a dataloader assuming the raw input loaded in the form of
    (label_string, text) 2-tuples. Parameters include:
      - data: list of (label_string, text) pairs
      - embedding_model: an LLM model that encodes text as vector sequences
      - text_pipeline: an LLM-specific tokenizer that tokenizes the raw input into (word piece) tokens
      - label_pipeline: simple mapping from label strings to integer ids
      - batch_size: batch size for the resulting dataloader
      - max_len: maximum sequence length; shorter strings will be padded, longer will be truncated
      - shuffle: flag for whether to shuffle input (e.g. for during training)
      - projection: sklearn projection object to reduce dimensionality of the embeddings
    """
    device = get_device()

    def collate_batch(batch):
        label_list, embedding_list, mask_list = [], [], []

        for (_label, _text) in batch:
            ## .. TODO ..
            ## 1. construct the label_list, embedding_list and mask_list elements for a data batch
            ## 2. the text_pipeline should be used to get tokenized text and the data mask
            ## 3. the embedding_model will take the tokenized result and produce and encoding/embedding of the input
            ## 4. finally, the projection (if provided - i.e. not None) should be used to reduce the
            ##    dimensionality of the embedding

            processed_text = text_pipeline(_text, max_length = max_len, padding = 'max_length', truncation = True,return_tensors="pt")
            input_ids, attention_mask = processed_text['input_ids'], processed_text['attention_mask']
            embedding = embedding_model(input_ids).last_hidden_state[0]
            if projection:
                embedding = torch.from_numpy(projection.transform(embedding.detach().numpy()))

            label_list.append(label_pipeline(_label))
            embedding_list.append(embedding)
            mask_list.append(attention_mask)

        label_list = torch.tensor(label_list, dtype=torch.int64)
        embedding_list = torch.stack(embedding_list)
        mask_list = torch.vstack(mask_list)
        return label_list.to(device), embedding_list.to(device), mask_list.to(device)

    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch)


def get_hf_loader_cached(data, batch_size, shuffle=False):
    """Construct a dataloader assuming the data is comming from a cache file
    in the form of (label_id, embedding, mask) triples.
    """
    device = get_device()

    def collate_batch_from_cache(batch):
        label_list, embedding_list, mask_list = [], [], []
        for (_label, _embedding, _mask) in batch:
            _emb = torch.Tensor(_embedding)
            label_list.append(_label)
            mask_list.append(torch.Tensor(_mask))
            embedding_list.append(_emb.unsqueeze(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        embedding_list = torch.vstack(embedding_list)
        mask_list = torch.vstack(mask_list)
        return label_list.to(device), embedding_list.to(device), mask_list.to(device)

    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch_from_cache)


def get_cached_data(json_l_file):
    """Read in a cached dataset within a JSON list file. Resulting data is a list of
    3-tuples of the form (label_id, embedding, mask)
    """
    data = []
    with io.open(json_l_file, 'rb') as fp:
        for line in fp:
            js_obj = orjson.loads(line)
            data.append((js_obj["label_id"], js_obj["embedding"], js_obj["mask"]))
    return data


def export_to_cache(dataloader, ofile):
    """Export the data within a dataloader to the target file in JSON list format.  Each JSON element has
    fields:
      - 'label_id' - the integer index for the data element's label
      - 'embedding' - the tensor (vector sequence) representation of the input
      - 'mask'- the vector indicating which positions are valid inputs in the input sequence (i.e. not padding)
    Note that this cache is implemented in a very simplified fashion without some fairly obvious optimizations,
    but should be sufficient to speed things up for this assignment.
    """
    with io.open(ofile, 'wb') as fp:
        for _, (label, data, mask) in enumerate(dataloader):
            for i in range(label.size()[0]):
                js_dd = {'label_id': int(label[i]),
                         'embedding': data[i].cpu().detach().numpy(),
                         'mask': mask[i].cpu().detach().numpy()}
                js_str = orjson.dumps(js_dd, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_APPEND_NEWLINE)
                fp.write(js_str)


def get_hf_data_loaders(train_file, val_file, test_file, batch_size, max_len, label_map,
                        # hf_llm_name='distilbert-base-uncased',
                        hf_llm_name='bert-base-uncased',
                        cache_dir=None,
                        proj_size=0):
    """Construct dataloaders for the training, validation and test data with fixed context sensitive
    embeddings from a HuggingFace language model.  Handle caching of the embeddings so that they
    need not be recomputed on each training run.
    """
    label_pipeline = lambda x: label_map.get(x, 0)

    cache_path = Path(cache_dir) if cache_dir else None
    if cache_dir and cache_path and (cache_path / 'train.json').exists():
        train_data = get_cached_data(cache_path / 'train.json')
        val_data = get_cached_data(cache_path / 'val.json')
        test_data = get_cached_data(cache_path / 'test.json')
        train_dataloader = get_hf_loader_cached(train_data, batch_size, shuffle=True)
        val_dataloader = get_hf_loader_cached(val_data, batch_size, shuffle=False)
        test_dataloader = get_hf_loader_cached(test_data, batch_size, shuffle=False)
    else:
        text_pipeline, embedding_model = get_llm(hf_llm_name)
        if proj_size > 0:
            projection = SparseRandomProjection(n_components=proj_size)
            d_out = embedding_model(**text_pipeline('Test string', return_tensors='pt')).last_hidden_state.size()[-1]
            projection.fit(np.random.randn(1, d_out))
        train_data = get_data(train_file)
        val_data = get_data(val_file)
        test_data = get_data(test_file)
        train_dataloader = get_hf_loader(train_data, embedding_model, text_pipeline, label_pipeline, batch_size,
                                         max_len,
                                         shuffle=True, projection=projection)
        val_dataloader = get_hf_loader(val_data, embedding_model, text_pipeline, label_pipeline, batch_size, max_len,
                                       shuffle=False, projection=projection)
        test_dataloader = get_hf_loader(test_data, embedding_model, text_pipeline, label_pipeline, batch_size, max_len,
                                        shuffle=False, projection=projection)
        if cache_dir:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            export_to_cache(train_dataloader, cache_path / 'train.json')
            export_to_cache(val_dataloader, cache_path / 'val.json')
            export_to_cache(test_dataloader, cache_path / 'test.json')

    return train_dataloader, val_dataloader, test_dataloader


llm_catalog = {
    'distilbert-base-uncased': (DistilBertTokenizer.from_pretrained, DistilBertModel.from_pretrained),
    'bert-base-uncased' : (AutoTokenizer.from_pretrained, BertModel.from_pretrained),
    ##'openai-gpt' : (AutoTokenizer.from_pretrained, OpenAIGPTModel.from_pretrained)
    ## add more model options here if desired
}


def get_llm(model_name):
    tok_fn, model_fn = llm_catalog[model_name]
    return tok_fn(model_name), model_fn(model_name)
