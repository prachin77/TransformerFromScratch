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
import sys

# Fix Windows console encoding for non-Latin scripts (e.g. Hindi)
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
# if hf_token:
#     login(token=hf_token, add_to_git_credential=False)
# else:
#     print("HF_TOKEN not found; using unauthenticated Hugging Face access.")

def greedy_decode(model , source , source_mask , tokenizer_src , tokenizer_tgt , max_len , device , repetition_penalty : float = 0.0):
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
        # old line kept here because it's still the correct/clean greedy decode :
        # _ , next_word = torch.max(prob , dim = 1)

        # why this penalty ?
        # greedy decode always takes the single most likely token . when the model is
        # still undertrained its most likely token is just the most frequent word , so it
        # keeps producing the same word again and again ("और और और ..." , ", , , ,")
        # until it reaches max_len , because it never predicts [EOS] .
        # so before picking the max we pull down the score of every token that is already
        # inside decoder_input . this does NOT make the model learn better , it only makes
        # the validation print readable while we're still training .
        # once the model starts predicting [EOS] by itself set "repetition_penalty" : 0.0
        # in config.py and you're back to plain greedy (the old line above).
        if repetition_penalty > 0.0:
            for tok in set(decoder_input[0].tolist()):
                # don't touch SOS , the first step has nothing else to work with
                if tok != sos_idx:
                    prob[0 , tok] -= repetition_penalty

        _ , next_word = torch.max(prob , dim = 1)
        decoder_input = torch.cat([decoder_input , torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)] , dim = 1)

        if next_word == eos_idx:
            break

    # we squeeze it to remove the batch dim 
    return decoder_input.squeeze(0)


def run_validation(model , validation_ds , tokenizer_src , tokenizer_tgt , max_len , device , print_msg , global_state , writer , num_examples = 2 , repetition_penalty : float = 0.0):
    # num_examples = 2 -> means that we'll testcase on 2 sentences and see how they got translated
    # to visualize output of the model while we're training.
    # just to see how our model is evolving while its training.
    # it'll also allow to inference the model and lets us see how they got translated

    # inference only , so no autocast and no gradients here .
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

            model_out = greedy_decode(model , encoder_input , encoder_mask , tokenizer_src , tokenizer_tgt , max_len , device , repetition_penalty)

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
    # old line was : if not Path.exists(tokenizer_path):
    # problem -> the tokenizer jsons committed in the repo already exist and have 30k
    # words each . so this function silently loaded them and never built anything against
    # the data we are training on right now . with the "test" split that meant 30k output
    # classes for ~450 sentences , which is exactly why the model only ever printed the
    # most frequent token . "rebuild_tokenizer" in config.py lets us force a rebuild
    # without deleting the files by hand . leave it False normally .
    if (not Path.exists(tokenizer_path)) or config.get("rebuild_tokenizer" , False):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        # trainer = WordLevelTrainer(special_tokens=["[UNK]" , "[PAD]" , "[SOS]" , "[EOS]"] , min_frequency=2)
        # min_frequency = means that for a word to appear in our vocab it has to have min freq of atleast 2
        # added vocab_size so the vocab doesn't explode when we train on the 100k subset .
        # with min_frequency=2 alone , 100k sentences gave us ~80k words and the output
        # layer (d_model x vocab) just becomes big and slow , it doesn't help translation .
        trainer = WordLevelTrainer(
            vocab_size = config.get("vocab_size" , 16000) ,
            special_tokens=["[UNK]" , "[PAD]" , "[SOS]" , "[EOS]"] ,
            min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(ds , lang) , trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_ds(config):
    # old version trained on the test split :
    # ds_raw = load_dataset(
    #     "Helsinki-NLP/opus-100",
    #     f'{config["lang_src"]}-{config["lang_tgt"]}',
    #     # split = "train" # keep this if you have compute and resources
    #     split = "test",
    #     token = hf_token
    # )
    #
    # why changed ?
    # the en-hi test split has only ~500 pairs , so training saw ~450 sentences (that's
    # the 225 batches x batch_size 2 in the logs) . no amount of epochs fixes that , the
    # model can only learn "which hindi word is most common" and keeps printing it .
    # so we now take the real train split and cap it with "train_samples" in config.py
    # so the first run on a T4 doesn't take hours . raise the cap or set it to None later .
    ds_raw = load_dataset(
        "Helsinki-NLP/opus-100",
        f'{config["lang_src"]}-{config["lang_tgt"]}',
        split = "train",
        token = hf_token
    )

    # keep only the first N pairs . this also means the tokenizers below get built on
    # this subset (only if they don't already exist , see get_or_build_tokenizer) .
    if config.get("train_samples"):
        # old version took a contiguous prefix :
        # ds_raw = ds_raw.select(range(min(config["train_samples"] , len(ds_raw))))
        #
        # why shuffle first ?
        # opus-100 is not shuffled , so rows 0..N come from whatever corpus / domain sits
        # at the top of the file . taking a prefix means our 100k pairs AND the tokenizer
        # vocab are both biased towards that domain , and that bias then shows up as
        # weird translations later . shuffling first gives us a random slice of the whole
        # file instead of a slice of the first domain .
        # seed = 42 keeps it reproducible , every run picks the exact same rows .
        # this only permutes indices in memory , it doesn't download anything extra .
        ds_raw = ds_raw.shuffle(seed=42).select(range(min(config["train_samples"] , len(ds_raw))))

    print(f"Total sentence pairs loaded : {len(ds_raw)}")

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
    # NOTE : this walks the whole (now much bigger) subset , on 100k pairs it takes a
    # minute or two before training even starts . that's normal , not a hang .
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


# ===== lr schedule =====
# the paper doesn't keep lr fixed . it raises it slowly for the first few thousand steps
# and then lets it fall again . our old fixed lr = 1e-3 with Adam had no warmup , which is
# why the loss was bouncing between 6.23 and 6.57 instead of going down .
#
# step 1      -> almost 0 lr (basically a no-op , the model takes tiny steps)
# step ~4000  -> peak lr (~1e-3 when d_model = 128 , it scales with d_model by itself)
# after that  -> lr decays like 1 / sqrt(step) , so late training is gentler
#
# we wrap the optimizer instead of torch's LambdaLR because the paper's formula uses
# d_model directly , and this way we keep the same optimizer object that gets saved
# inside the checkpoint .
class NoamOpt:
    def __init__(self , optimizer , d_model : int , warmup : int = 4000) -> None:
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup = warmup
        self.step_num = 0 # how many optimizer steps we've taken so far
        self.lr = 0.0 # lr that the current step used , just for logging in tensorboard

    def rate(self):
        self.step_num += 1
        return (self.d_model ** -0.5) * min(
            self.step_num ** -0.5 ,
            self.step_num * (self.warmup ** -1.5)
        )

    # old version , used to also do the optimizer step itself :
    # def step(self):
    #     self.lr = self.rate()
    #     for group in self.optimizer.param_groups:
    #         group["lr"] = self.lr
    #     # the 2 lines below were sitting inside the training loop before :
    #     #     optimizer.step()
    #     #     optimizer.zero_grad()
    #     # now they live here so the lr is always set right before the update .
    #     self.optimizer.step()
    #     self.optimizer.zero_grad()
    #
    # why split in 2 now ?
    # with fp16 the update has to go through GradScaler.step(optimizer) instead of
    # optimizer.step() (it unscales the gradients first and skips the step if any of them
    # turned into inf / nan) . so NoamOpt can no longer own the optimizer.step() call .
    # set_lr() only computes + applies the lr , the loop then calls
    # scaler.step(scheduler.optimizer) itself .
    def set_lr(self):
        self.lr = self.rate()
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr
        return self.lr


# ===== Training loop =======
def train_model(config):
    # define the device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    # fixed seed so the shuffled subset and the 90/10 split are the same on every run .
    # without it two runs on the same data can end up validating on different sentences .
    torch.manual_seed(config.get("seed" , 42))

    print(f"using device : {device}")

    # making sure that models folder exists or not 
    Path(config["model_folder"]).mkdir(parents=True , exist_ok=True)

    # load the dataset
    train_dataloader , val_dataloader , tokenizer_src , tokenizer_tgt = get_ds(config)
    # get vocab size
    model = get_model(config , tokenizer_src.get_vocab_size() , tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard
    writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(
        model.parameters() , 
        lr = config["lr"] , # this is only the starting lr now , NoamOpt overwrites it every step
        eps=1e-9,
        # disables Adam's multi-tensor implementation, which can use additional peak memory. 
        # It may run a little slower, but that is a reasonable trade-off for your GPU.
        foreach=False
    )

    # warmup + decay schedule , explained above the class . this is what actually makes
    # the lr change over training now , config["lr"] is basically ignored after step 1 .
    scheduler = NoamOpt(optimizer , config["d_model"] , config.get("warmup_steps" , 4000))

    # AMP = automatic mixed precision . a t4 is turing : it has fast fp16 tensor cores but
    # no bf16 and no tf32 , so a plain fp32 loop leaves most of the card idle .
    # autocast picks fp16 for the matmuls (where the speed is) and keeps softmax / layer
    # norm / the loss in fp32 (where fp16 would destroy the precision) .
    # GradScaler exists because fp16 can't represent very small gradients : it multiplies
    # the loss by a large number before backward and divides it back before the step .
    # turn it off with "use_amp" : False in config.py if you ever see nan / inf losses .
    use_amp = config.get("use_amp" , True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda" , enabled = use_amp)


    # this allows to restore the state of model and optimzer if in case model crashes in training
    initial_epoch = 0
    global_step = 0

    if config["preload"]:
        model_filename = get_weights_file_path(config , config['preload'])
        print(f"Preloading model : {model_filename}")
        state = torch.load(model_filename , map_location = device)
        initial_epoch = state['epoch'] + 1
        # why added ?
        # the old code restored the epoch , the optimizer and the global step but never
        # the weights themselves . so "resuming" actually continued training with a brand
        # new random model and a warmed up optimizer , which looks exactly like a stuck
        # loss . this one line is the actual resume .
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
        # keep the schedule where it left off , otherwise warmup restarts from 0
        scheduler.step_num = global_step

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
        ignore_index=tokenizer_tgt.token_to_id("[PAD]"),
        label_smoothing = 0.1 # takes 0.1% from the highest confident scores gives to others 
    ).to(device)

    for epoch in range(initial_epoch , config["epochs"]):
        # usually validation is ran after every few steps
        # but here i wanna see it at every iteration

        model.train()

        batch_iterator = tqdm(train_dataloader , desc = f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:
            # get the tensors
            encoder_input = batch["encoder_input"].to(device) # (B , seq_len)
            decoder_input = batch["decoder_input"].to(device) # (B , seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (B , 1 , 1 , seq_len) -> hides all the padding token
            decoder_mask = batch["decoder_mask"].to(device) # (B , 1 , seq_len , seq_len) -> hides all the padding token + also hides sibsequent words 

            # run the tensors through the transformer 
            # old version , plain fp32 , 3 separate calls :
            # encoder_output = model.encode(encoder_input , encoder_mask) # (B , seq_len , d_model)
            # decoder_output = model.decode(encoder_output , encoder_mask , decoder_input , decoder_mask) # (B , seq_len , d_model)
            # # projection output bcz we wanna map decoder output to the vocab
            # proj_output = model.project(decoder_output) # (B , seq_len , tgt_vocab_size)
            #
            # why one call now ?
            # we wrap the whole forward pass in a single autocast block so torch can run the
            # matmuls in fp16 and keep softmax / layer norm / the loss in fp32 by itself .
            # model(...) is just encode -> decode -> project in the same order (see
            # Transformer.forward in model.py) , so nothing about the math changed .
            # when use_amp is False the context manager is a no-op and this is plain fp32 again .
            with torch.autocast(device_type = "cuda" , dtype = torch.float16 , enabled = use_amp):
                proj_output = model(encoder_input , encoder_mask , decoder_input , decoder_mask) # (B , seq_len , tgt_vocab_size)

                # here we'll compare output of the model to the labels 
                # each dim tells whats the pos of each word in that vocab
                # also we want these 2 to be comparable
                # (B , seq_len , tgt_vocab_size) == (B , seq_len)
                label = batch["label"].to(device) # (B , seq_len)

                # it transforms 
                # (B , seq_len , tgt_vocab_size) -> (B * seq_len , tgt_vocab_size)
                # computing the loss inside autocast lets torch keep it in fp32 .
                loss = loss_fn(proj_output.view(-1 , tokenizer_tgt.get_vocab_size()) , label.view(-1))
            batch_iterator.set_postfix({f"loss" : f"{loss.item():6.3f}"})

            # log the loss in tensorboard
            writer.add_scalar("train loss : " , loss.item() , global_step)
            # also log the lr so we can actually see the warmup + decay in tensorboard
            writer.add_scalar("lr : " , scheduler.lr , global_step)

            # Backpropogate the loss 
            # old version , plain fp32 :
            # loss.backward()
            #
            # scheduler.step()
            # why the scaler ? fp16 gradients underflow to 0 very easily . scale() multiplies
            # the loss by a big factor before backward so they stay representable , then
            # step() divides it back , checks for inf/nan and only then calls optimizer.step() .
            # set_lr() has to happen before scaler.step() , it's what applies this step's lr .
            # the last 3 lines (step / update / zero_grad) are what scheduler.step() used to do
            # internally , they're split out here because the scaler owns the optimizer.step() .
            scaler.scale(loss).backward()
            scheduler.set_lr()
            scaler.step(scheduler.optimizer)
            scaler.update()
            scheduler.optimizer.zero_grad()

            global_step += 1

        run_validation(model , val_dataloader , tokenizer_src , tokenizer_tgt , config["seq_len"] , device , lambda msg : batch_iterator.write(msg) , global_step , writer , repetition_penalty = config.get("repetition_penalty" , 0.0))

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
