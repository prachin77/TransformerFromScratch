from pathlib import Path


def get_config():
    return {
        "batch_size" : 8,
        # "batch_size" : 2,
        "epochs" : 20,
        "lr" : 10**-3, # can vary according to you machine , dataset , no. of layers and other factors
        "seq_len" : 350, # can vary according to you machine , dataset , no. of layers and other factors
        "d_model" : 512,
        # "d_model" : 128,
        "lang_src" : "en",
        "lang_tgt" : "hi",
        "model_folder" : "weights",
        "model_basename" : "t_model",
        "preload" : None, # if we wanna restart the training after may be its crashed
        "tokenizer_file" : "tokenizer_{0}.json", # {0} for bcz it'll be saved for en and hi like tokenizer.en 
        "experiment_name" : "runs/model", # Tensorboard = saves losses and will help to visualize it

        # lr = 1e-3 fixed for the whole run made the loss bounce around ~6.2 on a T4
        # (that's roughly "predict the most frequent token" and nothing more).
        # the paper warms the lr up from 0 and then decays it , so warmup_steps is how
        # many steps that warmup lasts . 4000 is the paper's value and is fine here too ,
        # lower it to ~500 only if you train on a really small subset .
        "warmup_steps" : 4000,

        # the tokenizer jsons sitting in the repo already have 30k words each . as long
        # as the file exists get_or_build_tokenizer just loads it and never looks at the
        # data we're training on right now . flip this to True ONCE after you change the
        # dataset (and then back to False) so the vocab actually matches our corpus .
        "rebuild_tokenizer" : False,

        # vocab cap , only used when the tokenizer is actually being (re)built .
        # without a cap min_frequency=2 on 100k sentences gives ~80k words , which only
        # makes the projection layer huge and slow without helping translation .
        # 16k keeps the output layer small enough for a T4 .
        "vocab_size" : 16000,

        # the en-hi train split is big (538k pairs) and 1 epoch on a T4 would take hours .
        # we keep only this many pairs for now . 50k -> ~45k train / 5,625 steps per epoch ,
        # which is a few minutes per epoch with AMP . set it to None to use everything .
        "train_samples" : 50_000,

        # only used while validating . greedy decoding keeps repeating the same frequent
        # word when the model is undertrained , so we subtract this from tokens that were
        # already generated to force it to say something else . set it to 0.0 to turn it
        # off once the model starts predicting [EOS] on its own .
        "repetition_penalty" : 2.0,

        # mixed precision (fp16 + GradScaler) . on a t4 this is the single biggest speedup
        # because turing has fp16 tensor cores but NO bf16 / tf32 , so fp32 leaves the card
        # half idle . set it to False if you ever see nan/inf losses or you're debugging a
        # numerical problem .
        "use_amp" : True,

        # fixed seed . needed for 2 gpu training (both processes must build the same 90/10
        # split and the same shuffled subset) and handy for reproducing a run later .
        "seed" : 42
    }

def get_weights_file_path(config , epoch : str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)

