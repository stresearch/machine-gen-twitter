import argparse
import glob
import logging
from collections import defaultdict
import os

# os.environ["PYTORCH_TRANSFORMERS_CACHE"] = "/proj/sonar/huggingface"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
from argparse import Namespace
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import tqdm
from transformers.optimization import get_linear_schedule_with_warmup
from pandas.errors import ParserError
from deepspeed.ops.adam import FusedAdam
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict



def compute_metrics(preds, labels, loss):
    T = labels.shape[1]
    return dict(perplexity=np.exp(loss / T))


class FineTuned(pl.LightningModule):
    def __init__(self, lm_name="gpt2-medium", lr=5e-5, weight_decay=0.0, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.lm = AutoModelForCausalLM.from_pretrained(lm_name)
        if len(kwargs) == 0:
            print(f"unknown params {kwargs}")

    def forward(self, **inputs):
        return self.lm(**inputs)

    def _step(self, batch):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
        inputs["labels"] = batch[0].masked_fill(batch[1] == 0, -100)
        outputs = self(**inputs)
        loss = outputs[0]
        return dict(loss=loss)

    def training_step(self, batch, batch_idx):
        out = self._step(batch)
        for k, v in out.items():
            self.log(f"train_{k}", v)
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self._step(batch)
        for k, v in out.items():
            self.log(f"val_{k}", v)

    def configure_optimizers(self):
        optim_class = FusedAdam #torch.optim.AdamW

        optimizer = optim_class(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer


def get_data(model_name, split, dataset, field="text"):

    file_name = f"{dataset}_{split}_{model_name.replace('/','-')}.pt"

    if os.path.exists(file_name):
        dataset = torch.load(file_name)
    else:
        print("processing data", f"{dataset}_{split}.csv")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        try:
            data_test = pd.read_csv(f"{dataset}_{split}.csv")
        except ParserError:
            print("using python engine")
            data_test = pd.read_csv(f"{dataset}_{split}.csv", engine="python")

        texts = data_test.iloc[:, 0] if field is None else data_test.loc[:, field]
        texts = (
            tokenizer.eos_token + tokenizer.eos_token.join(texts) + tokenizer.eos_token
        )
        dataset = tokenizer(
            texts,
            stride=4,
            max_length=72,
            return_overflowing_tokens=True,
            truncation=True,
            return_tensors="pt",
            padding=True,
        )

        torch.save(dataset, file_name)
    fields = ["input_ids", "attention_mask"]
    return torch.utils.data.TensorDataset(*[dataset[f] for f in fields])


def generate(model, args, N):
    texts = []
    tokenizer = AutoTokenizer.from_pretrained(args.lm_name)
    tokenizer.pad_token = tokenizer.eos_token

    assert not model.lm.training 

    B = 10 * 5
    # B = 10
    with torch.no_grad():
        prompt = (
            tokenizer(tokenizer.eos_token, return_tensors="pt")["input_ids"]
            .to(model.device)
            .expand(B, -1)
        )

        for i in tqdm.trange(N // B):
            out = model.lm.generate(
                prompt,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                add_special_tokens=False,
                max_length=128,
                early_stopping=True,
                top_p=0.95,
                num_return_sequences=1,
            )

            # texts.append(tokenizer.decode(out[0].cpu(), skip_special_tokens=True))
            texts.extend(
                [tokenizer.decode(o.cpu(), skip_special_tokens=True) for o in out]
            )

    file_name = f"{args.dataset}_{args.lm_name.replace('/','-')}_mg_{N}.csv"

    pd.DataFrame(dict(text=texts)).to_csv(file_name, index=False)


def save_hg_model(source_checkpoint, destination_directory):

    # model_path = "/proj/semafor/kirill/covid/tune-gpt2-covid/722b7d4c1d8540319160c34cbca3860f/checkpoints/epoch=1-step=22475.ckpt"
    model_train = FineTuned.load_from_checkpoint(source_checkpoint)
    model = model_train.lm
    tokenizer = AutoTokenizer.from_pretrained(model_train.hparams.lm_name)
    model.config.task_specific_params = {}
    model.config.task_specific_params["text-generation"] = {
        "do_sample": True,
        "max_length": 128,
        "early_stopping": True,
        "top_p": 0.9,
    }

    # hg_model_name = "/proj/semafor/kirill/covid/gpt2-medium-covid"
    os.makedirs(destination_directory, exist_ok=True)
    model.save_pretrained(destination_directory)
    tokenizer.save_pretrained(destination_directory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm_name", default="EleutherAI/gpt-neo-1.3B")
    parser.add_argument("--dataset", default="avax")
    parser.add_argument("--mode", default="train")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_batch_size", type=int, default=16)
    parser.add_argument("--num_samples",type=int,default=1000 )

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    mode = args.mode.split(",")
    # path is None
    if "train" in mode:

        # pl.seed_everything(11)


        train_data = get_data(args.lm_name, "train", args.dataset)
        test_data = get_data(args.lm_name, "test", args.dataset)

        if "prep_data_only" in mode:
            return

        model = FineTuned(lm_name=args.lm_name, dataset=args.dataset)


        # return
        if args.model_batch_size < args.batch_size:
            accumulate_grad_batches = args.batch_size // args.model_batch_size
            print("accumulate_grad_batches: ", accumulate_grad_batches)
        else:
            accumulate_grad_batches = None

        train_dataloader = torch.utils.data.DataLoader(
            train_data, shuffle=True, batch_size=args.model_batch_size, pin_memory=True
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_data,
            shuffle=False,
            batch_size=args.model_batch_size * 4,
            pin_memory=True,
        )

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_loss", mode="min"
        )
        # gpu = 3

        trainer = pl.Trainer(
            gpus=8,  # comment this line for multi-gpu
            strategy="deepspeed_stage_2",
            precision=16,
            min_epochs=1,
            max_epochs=5,
            # limit_train_batches = .1,
            # limit_val_batches = .1,
            accumulate_grad_batches=accumulate_grad_batches,
            callbacks=[
                # pl.callbacks.EarlyStopping(monitor="val_loss"),
                checkpoint_callback,
            ],
        )

        trainer.fit(model, train_dataloader, test_dataloader)

        path = checkpoint_callback.best_model_path

    if "generate" in mode:

        path = find_checkpoint(args)
        # print(path)
        # return

        model = FineTuned.load_from_checkpoint(path, strict=False)
        model.lm.tie_weights()
        model.to(f"cuda:{args.gpu}")

        generate(model, args, args.num_samples)

    ## save some generations

def find_checkpoint(args, search_path = "lightning_logs"):
    for p in glob.glob(os.path.join(search_path,"*")):
        ckpt_path = get_checkpoint(p,args)
        if ckpt_path is not None:
            return ckpt_path


def get_checkpoint(model_dir,args):
    import yaml
    yaml_file = os.path.join(model_dir, "hparams.yaml")
    params = yaml.unsafe_load(open(yaml_file))
    if params.get("dataset", None) == args.dataset and params["lm_name"] == args.lm_name:
        #check for regular checkpoint first
        ckpt_path = glob.glob(os.path.join(model_dir,"checkpoints","*.ckpt"))
        if len(ckpt_path)>0 and not os.path.isdir(ckpt_path[0]):
            print("found regular checkpoint")
            return ckpt_path[0]


        #check for combined model first
        ckpt_path = glob.glob(os.path.join(model_dir,"checkpoints","*","lightning_model.pt"))
        if len(ckpt_path) > 0:
            print("combined checkpoint found", ckpt_path[0])
            return ckpt_path[0]

        ckpt_path = glob.glob(os.path.join(model_dir,"checkpoints","*.ckpt"))
        if len(ckpt_path) > 0:
            print("distr checkpoint found.. combining", ckpt_path[0])
            path = ckpt_path[0]
            output_path = os.path.join(path,"lightning_model.pt")
            convert_zero_checkpoint_to_fp32_state_dict(path, output_path)
            return output_path






if __name__ == "__main__":
    main()
