from pathlib import Path


def get_config():
    return {
        "batch_size" : 8,
        "epochs" : 20,
        "lr" : 10**-3, # can vary according to you machine , dataset , no. of layers and other factors
        "seq_len" : 350, # can vary according to you machine , dataset , no. of layers and other factors
        "d_model" : 512,
        "lang_src" : "en",
        "lang_tgt" : "hi",
        "model_folder" : "weights",
        "model_basename" : "t_model",
        "preload" : None, # if we wanna restart the training after may be its crashed
        "tokenizer_file" : "tokenizer_{0}.json", # {0} for bcz it'll be saved for en and hi like tokenizer.en 
        "experiment_name" : "runs/model" # saves losses
    }

def get_weights_file_path(config , epoch : str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)

