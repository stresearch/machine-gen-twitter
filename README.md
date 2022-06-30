# Instructions for Constructing a Dataset of Social Media Machine Generated Text 

This project contains code for constructing a dataset for development and evaluation of forensic tools for detecting machine generated text in social media.

There are 4 steps:
1. Rehydrate source data from tweet ids
2. Fine-tune natural language generation (NLG) models
3. Generate machine generated text from NLG models and construct human, machine and mixed timelines of tweets

## Rehydrate Source Data

We source data from 3 english language Twitter datasets on 3 different topics:
- Vaccine: mostly anti-vax discussion around vaccines from https://github.com/gmuric/avax-tweets-dataset
- COVID: general covid dicussion from around the world from [TBD]. This dataset was specifically constructed to have a significant number of users with more than 1 tweets.
- Climate: general climate change discussion https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/5QCCUU

To rehydrate a tweet bodies from tweet_ids use `rehydrate` function in `utils.py`. Twitter. The output is saved as `.jsonl`; each row is a tweet. We keep 100k samples from each dataset

## Fine-tune NLG Models

The next step is to fine several NLG models. 
### Preprocess data for training

We preprocess each dataset by:
- Cleaning up tweet formatting
- Droppping `RT` and `@mentions` for retweets
- Dropping duplicates
- Dropping boring tweets i.e. tweets with less than 5 non-trivial (borrowed from [huggingtweets](https://github.com/borisdayma/huggingtweets))

We split the clean data it into train and validation sets (95%/5%) to monitor model training for early stopping. 

Refer to `create_dataset` function in `utils.py`

### Fine-tune

We fine-tune 4 different pre-trained NLG models for each dataset: `gpt2, gpt2-medium, gpt2-large,EleutherAI-gpt-neo-1.3B` . We use pre-trained models from huggingface transformers: https://huggingface.co/models

To fine-tune: 
```shell
python train_deepspeed.py --help
usage: train_deepspeed.py [-h] [--lm_name LM_NAME] [--dataset DATASET] [--mode MODE] [--gpu GPU] [--batch_size BATCH_SIZE] [--model_batch_size MODEL_BATCH_SIZE]
                          [--num_samples NUM_SAMPLES] [--strategy STRATEGY] [--max_epochs MAX_EPOCHS]

optional arguments:
  -h, --help            show this help message and exit
  --lm_name LM_NAME     huggingface model name (default: EleutherAI/gpt-neo-2.7B)
  --dataset DATASET     dataset name (default: avax)
  --mode MODE           mode = train,generate (default: train)
  --gpu GPU             gpus to use (default: 0)
  --batch_size BATCH_SIZE
                        desired total batch size (default: 32)
  --model_batch_size MODEL_BATCH_SIZE
                        batch that fits on gpu (default: 2)
  --num_samples NUM_SAMPLES
                        number of samples to generate (default: 1000)
  --strategy STRATEGY   model parallelization strategy, use deepspeed_2 or 3 for large models to shard (default: None)
  --max_epochs MAX_EPOCHS
                        max epochs (default: 5)

```

We concatenate all tweet text with `EOS_TOKEN` and split it into chunks of max_lenght of 72 with  overlap of 4 tokens. We use effective size of 32 and train for 5 epochs with learning rate of 5e-5 optimzing the causal language model objective i.e. next word prediciton. We keep the model with best loss on the validation set.

## Generate and construct timelines
