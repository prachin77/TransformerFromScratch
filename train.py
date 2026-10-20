import torch
import torch.nn as nn
from model import build_transformer
from torch.utils.data import Dataloader , Dataset , random_split
from dataset import BilingualDataset , casual_mask
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path


def get_all_sentences(ds , lang):
    for item in ds:
        yield item["transaltion"][lang] # extracts only one language from pair of sentences in ds   

def get_or_build_tokenizer(config , ds , lang):
    # config = configuration of our model
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel)(unk_token = "[UNK]")
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]" , "[PAD]" , "[SOS]" , "[EOS]"] , min_frequency=2)
        # min_frequency = means that for a word to appear in our vocab it has to have min freq of atleast 2
        tokenizer.train_from_iterator(get_all_sentences(ds , lang) , trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_ds(config):
    ds_raw = load_dataset(
        "Helsinki-NLP/opus-100",
        f'{config["lang_src"]}-{config["lang_tgt"]}',
        split = "train"
    )

    # build tokenizers
    tokenizer_src = get_or_build_tokenizer(config , ds_raw , config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config , ds_raw , config["lang_tgt"])

    # 90% for training & 10% for validation 
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    # random_split splits the dataset based on the size we give in input 
    train_ds_raw , val_ds_raw = random_split(ds_raw , [train_ds_size , val_ds_size])

    train_ds = BilingualDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"]
    )
    val_ds = BilingualDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"]
    )

    # for chosing max seq_len , we need to watch what is the max len of each sentence 
    # in src & tgt for each of the 2 splits created here , so that if we chose very small
    # seq len we'll know
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        # load sentence from each pair i.e. src & tgt and then convert them into ids
        # using tokenizer and i check the length .
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_src.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src , len(src_ids))
        max_len_tgt = max(max_len_tgt , len(tgt_ids))

    print(f"Max length of source sentence : {max_len_src}")
    print(f"Max length of target sentence : {max_len_tgt}")

    # create datalolader 
    train_dataloader = Dataloader(train_ds , batch_size = config["batch_size"], shuffle = True)
    val_dataloader = Dataloader(val_ds , batch_size = 1 , shuffle = True)

    return train_dataloader , val_dataloader , tokenizer_src , tokenizer_tgt


def get_model(config , vocab_src_len , vocab_tgt_len):
    model = build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config["seq_len"], # src seq_len 
        config["seq_len"], # tgt seq_len
        config["d_model"]
    )

    return model


