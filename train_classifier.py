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
from tenacity import AttemptManager
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import tqdm
from transformers.optimization import get_linear_schedule_with_warmup


def compute_metrics(preds, labels, loss):
    T = labels.shape[1]
    return dict(perplexity=np.exp(loss / T))


class FineTuned(pl.LightningModule):
    def __init__(
        self,
        lm_name="roberta-base",
        lr=5e-5,
        weight_decay=0.0,
        num_labels = 2,
        input_dropout = .1
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lm = AutoModelForSequenceClassification.from_pretrained(lm_name, num_labels = num_labels)

    def forward(self, **inputs):
        return self.lm(**inputs)

    def random_mask(self, mask):
        if self.training:
            mask = mask*torch.bernoulli(torch.ones_like(mask)*(1.-self.hparams.input_dropout))
        return mask
        

    def _step(self, batch):
        batch[1] = self.random_mask(batch[1])
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
        outputs = self(**inputs)
        loss = outputs.loss
        acc = (outputs.logits.argmax(-1) == inputs["labels"]).float().mean()
        return dict(loss=loss,acc =acc)

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
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer


def get_data(model_name, split, dataset):

    file_name = f"{dataset}_{split}_{model_name}.pt"

    if os.path.exists(file_name):
        dataset = torch.load(file_name)
    else:

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # tokenizer.pad_token = tokenizer.eos_token
        data = pd.read_csv(f"{dataset}_{split}.csv")
        
        dataset = tokenizer(
            data["text"].values.tolist(),
            max_length=72,
            truncation=True,
            return_tensors="pt",
            padding=True,
        )
        dataset["labels"] = torch.from_numpy(data["is_misinfo"].astype(int).values)
        torch.save(dataset, file_name)
    fields = ["input_ids", "attention_mask","labels"]
    return torch.utils.data.TensorDataset(*[dataset[f] for f in fields])


def generate(model, args, N):
    texts = []
    tokenizer = AutoTokenizer.from_pretrained(args.lm_name)
    tokenizer.pad_token = tokenizer.eos_token
    with torch.no_grad():
        prompt = tokenizer(tokenizer.eos_token, return_tensors="pt")["input_ids"].to(
            model.device
        )

        for i in tqdm.trange(N):
            out = model.lm.generate(
                prompt,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                add_special_tokens=False,
                max_length=128,
                early_stopping=True,
                top_p=0.9,
                num_return_sequences=1,
            )

            texts.append(tokenizer.decode(out[0].cpu(), skip_special_tokens=True))

    file_name = f"{args.dataset}_{args.lm_name}_mg.csv"

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
    parser.add_argument("--lm_name", default="roberta-base")
    parser.add_argument("--dataset", default="vaccine")
    parser.add_argument("--mode", default="train")
    parser.add_argument("--batch_size", default=128)

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    mode = args.mode.split(",")
    # path is None
    if "train" in mode:

        # pl.seed_everything(11)

        model = FineTuned(lm_name=args.lm_name)

        train_data = get_data(args.lm_name, "train", args.dataset)
        test_data = get_data(args.lm_name, "test", args.dataset)

        # return

        train_dataloader = torch.utils.data.DataLoader(
            train_data, shuffle=True, batch_size=args.batch_size, pin_memory=True
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_data, shuffle=False, batch_size=args.batch_size, pin_memory=True
        )

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_loss", mode="min"
        )

        trainer = pl.Trainer(
            gpus=[5],  # comment this line for multi-gpu
            precision=16,
            min_epochs=1,
            max_epochs=20,
            # limit_train_batches = .1,
            # limit_val_batches = .1,
            # accumulate_grad_batches=2,
            callbacks=[
                # pl.callbacks.EarlyStopping(monitor="val_loss"),
                checkpoint_callback,
            ],
        )

        trainer.fit(model, train_dataloader, test_dataloader)

        path = checkpoint_callback.best_model_path

    # if "generate" in mode:
    #     # path = "lightning_logs/version_0/checkpoints/epoch=0-step=8.ckpt"
    #     model = FineTuned.load_from_checkpoint(path)
    #     model.to("cuda:7")

    #     generate(model, args, 1000)

    ## save some generations


if __name__ == "__main__":
    main()
