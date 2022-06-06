from attr import dataclass
import regex
import torch
import numpy as np
import pandas as pd
import tqdm
import glob
import os

def detect(model, tokenizer, texts, labels, device):

    data_tokenized = tokenizer(texts, max_length=512, truncation="only_first", padding=True, return_tensors="pt")


    feature_names = ["input_ids", "attention_mask"]

    data_set = torch.utils.data.TensorDataset(
        *[data_tokenized[f] for f in feature_names])
    data_loader = torch.utils.data.DataLoader(
        data_set, batch_size=128, shuffle=False, pin_memory=True)

    phat = []
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader):
            out = model(**{f: b.to(device)
                        for f, b in zip(feature_names, batch)})
            phat.append(out.logits.softmax(-1).cpu())

    phat = torch.cat(phat)


    correct = (phat.argmax(-1).numpy() == labels.astype(int))
    accuracy = correct.mean()

    scores = (phat[:,1] / phat[:,0]).log().numpy()

    return accuracy, scores


def make_longer(d,N = 2):
    sep = ". "
    temp = d.clean_text.values + sep
    temp = temp.reshape(len(temp)//N,-1).sum(-1)
    temp = pd.DataFrame(dict(clean_text = temp))
    temp.clean_text = temp.clean_text.str.slice(0,-len(sep))
    return temp

def load_dataset(file_name, expand = 1):
    data_real = pd.read_csv(file_name , engine="python").iloc[:1000]["text"].fillna("").str.replace("...","",regex=False)
    data_real.name = "clean_text"

    data_real = data_real.to_frame()
    if expand > 1:
        data_real = make_longer(data_real,expand)

    return data_real


# def combine(data_real,data_gen):
#     data_real["label"] = 1
#     data_gen["label"] = 0
#     data =pd.concat([data_real, data_gen], ignore_index=True)
#     return data


def get_detector(device):
    from transformers import RobertaForSequenceClassification, RobertaTokenizerFast


    detector_model_name = "roberta-large-openai-detector"

    # device = "cuda:1"
    tokenizer = RobertaTokenizerFast.from_pretrained(detector_model_name)
    model =     RobertaForSequenceClassification.from_pretrained(
            detector_model_name).eval().requires_grad_(False).to(device)

    return model, tokenizer

def eval_hk(input_dir ="/proj/semafor/hackathons/hk3/cp4/v1/task1_fixed/" ,
           result_dir = "/proj/semafor/hackathons/hk3/results/cp4/cp4_task1_openai-baseline", summary_file = None, device = "cuda:3"):
    model,tokenizer = get_detector(device)
    
    files = glob.glob(os.path.join(input_dir,"test*.json"))[::-1]
    sep = ". "

    print("total files ", len(files))

    outs = []

    for file in files:
        
        data = pd.read_json(file,orient="records")["tweets"].apply(lambda a: sep.join(a)).str.replace("...","",regex=False).fillna("")

        if "human" in file:
            labels = np.ones(len(data))
        else:
            labels = np.zeros(len(data))

        acc, scores_gen = detect(model, tokenizer, data.tolist(), labels, device)
        head, tail = os.path.split(file)

        outs.append(dict(file_name = tail, acc = acc))

        print("testing ",file)
        print("acc", acc)

        scores_gen = pd.DataFrame(scores_gen, columns = ["llr"])
        scores_gen.to_csv(os.path.join(result_dir,tail.replace(".json",".csv")),index=False)
    
    outs = pd.DataFrame(outs)
    outs.to_csv(summary_file)

def main(device = "cuda:4"):
    

    model,tokenizer = get_detector(device)


    datasets = ["avax","climate","graphika"]
    expand = [1, 5, 10, 20]
    #/proj/semafor/kirill/experiments/semafor/twitter_nlg/avax_EleutherAI-gpt-neo-1.3B_mg.csv
    models = ["gpt2-medium","gpt2","gpt2-large", "EleutherAI-gpt-neo-1.3B"]

    outs = []
    for d in datasets:
        for n in expand:
            print(d,n)
            data_real = load_dataset(f"{d}_test.csv",n)
            data_real["label"] = 1
            acc_real,scores_real = detect(model, tokenizer, data_real.clean_text.tolist(), data_real.label.values, device)
            outs.append(dict(acc = acc_real, dataset = d, num_tweets = n, model = "human"))


            for model_name in models:
            
                data_gen = load_dataset(f"{d}_{model_name}_mg.csv",n)
                data_gen["label"] = 0
                acc_gen,scores_gen = detect(model, tokenizer, data_gen.clean_text.tolist(), data_gen.label.values, device)
                outs.append(dict(acc = acc_gen, dataset = d, num_tweets = n, model = model_name))


    outs = pd.DataFrame(outs)
    outs.to_csv("detection_results_v2.csv")

if __name__=="__main__":

    eval_hk(input_dir ="/proj/semafor/hackathons/hk3/cp4/v1/task1_fixed/", 
            result_dir="/proj/semafor/hackathons/hk3/results/cp4/cp4_task1_openai-baseline",
            summary_file = "detection_results_hk_task1.csv")

    eval_hk(input_dir ="/proj/semafor/hackathons/hk3/cp4/v1/task2_fixed/",
            result_dir   = "/proj/semafor/hackathons/hk3/results/cp4/cp4_task2_openai-baseline",
            summary_file = "detection_results_hk_task2.csv")





