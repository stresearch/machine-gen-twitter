
import pandas as pd
import re
import numpy as np

def clean_tweet(tweet, allow_new_lines = False):
    tweet = tweet.replace('&amp;', '&')
    tweet = tweet.replace('&lt;', '<')
    tweet = tweet.replace('&gt;', '>')
    bad_start = ['http:', 'https:']
    for w in bad_start:
        tweet = re.sub(f" {w}\\S+", "", tweet)      # removes white space before url
        tweet = re.sub(f"{w}\\S+ ", "", tweet)      # in case a tweet starts with a url
        tweet = re.sub(f"\n{w}\\S+ ", "", tweet)    # in case the url is on a new line
        tweet = re.sub(f"\n{w}\\S+", "", tweet)     # in case the url is alone on a new line
        tweet = re.sub(f"{w}\\S+", "", tweet)       # any other case?
    tweet = re.sub(' +', ' ', tweet)                # replace multiple spaces with one space
    if not allow_new_lines:                         # TODO: predictions seem better without new lines
        tweet = ' '.join(tweet.split("\n"))
    return tweet.strip()


def drop_mentions(tweet):
    words = tweet.split(" ")
    return " ".join([w for w in words if not w.startswith("@")]).strip()

def get_length(tweet):
    return len(tweet.split(" "))

def boring_tweet(tweet, min_non_boring_words = 5):
    boring_stuff = ['http', '@', '#']
    not_boring_words = len([None for w in tweet.split() if all(bs not in w.lower() for bs in boring_stuff)])
    return not_boring_words < min_non_boring_words


def create_dataset(file_name, to_drop_mentions = True):
    # assume ends iwth jsonl
    if file_name.endswith(".jsonl"):
        data = pd.read_json(file_name,lines = True).fillna("")
        file_name = file_name.replace(".jsonl","")
    elif file_name.endswith(".csv"):
        data = pd.read_csv(file_name).fillna("")
        file_name = file_name.replace(".csv","")

    text = data.text.str.replace("RT ","").drop_duplicates()
    text = text.apply(clean_tweet)

    if to_drop_mentions:
        text = text.apply(drop_mentions)

    boring = text.apply(boring_tweet)

    text = text.loc[~boring].fillna("")

    print("length 95 percentile", text.apply(get_length).describe([.95]))
    print("length", len(text))

    #split test train
    train_mask = np.random.rand(len(text))<.9
    test_mask = ~train_mask

    file_name = file_name.replace(".jsonl","")

    text.loc[train_mask].to_csv(f"{file_name}_train.csv", index = False)
    text.loc[test_mask].to_csv(f"{file_name}_test.csv", index = False)


 

if __name__ == "__main__":
    create_dataset("graphika.csv")