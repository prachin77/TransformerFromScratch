from model import build_transformer
from dataset import BilingualDataset , casual_mask
from config import get_config , get_weights_file_path

import torch
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader , random_split
from torch.utils.tensorboard import SummaryWriter

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import warnings
from tqdm import tqdm
from pathlib import Path
import os


def greedy_decode(model , source , source_mask , tokenizer_src , tokenizer_tgt , max_len , device):
    # when we wanna inference the model we only calculate encoder output only once 
    # and reuse it for every token that model will output from decoder 
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    encoder_output = model.encode(source , source_mask)

    # how we do inferncing ?
    # first we give decoder SOS token so that it can output first token of the translated sentence .
    # then at every iteration we add previous token to the decoder input so that decoder can output
    # the next token . 

    # initialize decoder input with sos token 
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for the target (decoder input)
        # we don't want the input to watch future words
        decoder_mask = casual_mask(decoder_input.size(1)).type_as(source).to(device)

        # calculate output of the decoder 
        # reuse output of the encoder for every iteration
        out = model.decode(encoder_output , source_mask , decoder_input , decoder_mask)

        # get the next token
        prob = model.project(out[: , -1])
        # select the token with max prob 
        _ , next_word = torch.max(prob , dim = 1)
        decoder_input = torch.cat([decoder_input , torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)] , dim = 1)

        if next_word == eos_idx:
            break

    # we squeeze it to remove the batch dim 
    return decoder_input.squeeze(0)


def run_validation(model , validation_ds , tokenizer_src , tokenizer_tgt , max_len , device , print_msg , global_state , writer , num_examples = 2):
    # num_examples = 2 -> means that we'll testcase on 2 sentences and see how they got translated
    # to visualize output of the model while we're training.
    # just to see how our model is evolving while its training.
    # it'll also allow to inference the model and lets us see how they got translated

    model.eval()
    count = 0

    # source_texts = []
    # expected = []
    # predicted = []

    # size of the control window (just use the default value)
    console_width = 80

    with torch.no_grad(): 
        # disables gradient calculation during inferencing , 
        # it is enable only during training as we want the model to learn and calculate steepest descent

        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) 
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1 , "Batch size must be 1 for validation"

            model_out = greedy_decode(model , encoder_input , encoder_mask , tokenizer_src , tokenizer_tgt , max_len , device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            # source_texts.append(source_text)
            # expected.append(target_text)
            # predicted.append(model_out_text)

            # print to console . we'll not use simple print bcz we've progress bar also . 
            # so its reccommeded to use tqdm's print method during progress bar otherwise
            # if use use simple print the output will interfer when the progress bar is running .
            print_msg("-" * console_width)
            print_msg(f"SOURCE : {source_text}")
            print_msg(f"TARGET : {target_text}")
            print_msg(f"PREDICTED : {model_out_text}")

            if count == num_examples:
                break

    # if writer:
        # TorchMetrics , charerrorrate , BLUE , worderrorrate



def get_all_sentences(ds , lang):
    for item in ds:
        yield item["translation"][lang] # extracts only one language from pair of sentences in ds   

def get_or_build_tokenizer(config , ds , lang):
    # config = configuration of our model
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
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
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src , len(src_ids))
        max_len_tgt = max(max_len_tgt , len(tgt_ids))

    print(f"Max length of source sentence : {max_len_src}")
    print(f"Max length of target sentence : {max_len_tgt}")

    # create datalolader 
    train_dataloader = DataLoader(train_ds , batch_size = config["batch_size"], shuffle = True)
    val_dataloader = DataLoader(val_ds , batch_size = 1 , shuffle = True)

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


# ===== Training loop =======
def train_model(config):
    # define the device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(f"using device : {device}")

    # making sure that models folder exists or not 
    Path(config["model_folder"]).mkdir(parents=True , exist_ok=True)

    # load the dataset
    train_dataloader , val_dataloader , tokenizer_src , tokenizer_tgt = get_ds(config)
    # get vocab size
    model = get_model(config , tokenizer_src.get_vocab_size() , tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard
    writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters() , lr = config["lr"] , eps=1e-9)


    # this allows to restore the state of model and optimzer if in case model crashes in training
    initial_epoch = 0
    global_step = 0

    if config["preload"]:
        model_filename = get_weights_file_path(config , config['preload'])
        print(f"Preloading model : {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimzer_state_dict'])
        global_step = state["global_step"]

    # we need to tell what index to ignore . we want it to
    # ignore PAD token so that it doesn't contribute to the loss
    # We'll also use label smoothing 
    # label smoothing = it allows the model to be less confident about its descision
    # lets say you have 3 factors for classification and you want your model to give class to 
    # each one of them . ideally what model will do is it'll give 1 to correct class and 0 to 
    # incorrect ones which translates to 100% confidence 
    #       scores
    # class
    #     a 1 0 0 
    #     b 0 1 0
    #     c 0 0 1
    # we don't want this we want the model to give scores to even the incorrect one and 
    # reduce the confidence for the correct one . 
    # maybe -> this helps in reducing overfitting -> yeah my anticipation is true. hurrrrrrrayyyyyyyyy!
    #          scores
    #     class
    #         a 0.95 0.05 0.05 
    #         b 0.05 0.95 0.05
    #         c 0.05 0.05 0.95
    loss_fn = nn.CrossEntropyLoss(
        # ignore_index=tokenizer_tgt.token_to_id("[PAD]"),
        ignore_index=tokenizer_src.token_to_id("[PAD]"),
        label_smoothing = 0.1 # takes 0.1% from the highest confident scores gives to others 
    ).to(device)

    for epoch in range(initial_epoch , config["epochs"]):
        # usually validation is ran after every few steps
        # but here i wanna see it at every iteration

        # model.train()

        batch_iterator = tqdm(train_dataloader , desc = f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:
            model.train()

            # get the tensors
            encoder_input = batch["encoder_input"].to(device) # (B , seq_len)
            decoder_input = batch["decoder_input"].to(device) # (B , seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (B , 1 , 1 , seq_len) -> hides all the padding token
            decoder_mask = batch["decoder_mask"].to(device) # (B , 1 , seq_len , seq_len) -> hides all the padding token + also hides sibsequent words 

            # run the tensors through the transformer 
            encoder_output = model.encode(encoder_input , encoder_mask) # (B , seq_len , d_model)
            decoder_output = model.decode(encoder_output , encoder_mask , decoder_input , decoder_mask) # (B , seq_len , d_model)
            # projection output bcz we wanna map decoder output to the vocab
            proj_output = model.project(decoder_output) # (B , seq_len , tgt_vocab_size)

            # here we'll compare output of the model to the labels 
            # each dim tells whats the pos of each word in that vocab
            # also we want these 2 to be comparable
            # (B , seq_len , tgt_vocab_size) == (B , seq_len)
            label = batch["label"].to(device) # (B , seq_len)

            # it transforms 
            # (B , seq_len , tgt_vocab_size) -> (B * seq_len , tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1 , tokenizer_tgt.get_vocab_size()) , label.view(-1))
            batch_iterator.set_postfix({f"loss" : f"{loss.item():6.3f}"})

            # log the loss in tensorboard
            writer.add_scalar("train loss : " , loss.item() , global_step)

            # Backpropogate the loss 
            loss.backward()

            # update the weights
            optimizer.step()
            optimizer.zero_grad()

            run_validation(model , val_dataloader , tokenizer_src , tokenizer_tgt , config["seq_len"] , device , lambda msg : batch_iterator.write(msg) , global_step , writer)

            global_step += 1

        # save the model and the end of every epoch 
        model_filename = get_weights_file_path(config , f"{epoch:02d}")
        torch.save({
            "epoch" : epoch,
            "model_state_dict" : model.state_dict(),
            "optimizer_state_dict" : optimizer.state_dict(),
            "global_step" : global_step
        } , model_filename)

if __name__ == "__main__":
    # use this if 
    # warnings.filterwarnings("ignore")

    warnings.simplefilter("default")

    config = get_config()
    train_model(config)
