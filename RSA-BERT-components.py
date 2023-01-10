import numpy as np
import pandas as pd
import torch
import nltk
import transformers as ppb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Location of SST2 sentiment dataset
SST2_LOC = 'https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv'
WEIGHTS = 'distilbert-base-uncased'
# Define the size of the set as to reduce total training time
SET_SIZE = 2000

df2 = pd.read_csv("/Users/ericgulottyjr/Downloads/redditdata.csv")
test_df = pd.read_csv("/Users/ericgulottyjr/PoliticalDiscussion_HOTsubreddit.csv")

con = df2[12050:]
lib = df2[:804]
combined = [lib, con]
df3 = pd.concat(combined)

#df3
print(df3['Political Lean'].value_counts())

# Download the dataset from its Github location, return as a Pandas dataframe
def get_dataframe():
    df = pd.read_csv("path_to_csv")
    return df

# Extract just the labels from the dataframe
def get_labels(df):
    return df["Political Lean"]

# Get a trained tokenizer for use with BERT
def get_tokenizer():
    return ppb.DistilBertTokenizer.from_pretrained(WEIGHTS)

# Convert the sentences into lists of tokens
def get_tokens(dataframe, tokenizer):
    return dataframe["Title"].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

# We want the sentences to all be the same length; pad with 0's to make it so
def pad_tokens(tokenized):
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)
    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
    return padded

# Grab a trained DistiliBERT model
def get_model():
    return ppb.DistilBertModel.from_pretrained(WEIGHTS)

# Get model with get_model(), 0-padded token lists with pad_tokens() on get_tokens().
# Only returns the [CLS] vectors representing the whole sentence, corresponding to first token.
def get_bert_sentence_vectors(model, padded_tokens):
    # Mask the 0's padding from attention - it's meaningless
    mask = torch.tensor(np.where(padded_tokens != 0, 1, 0))
    with torch.no_grad():
        word_vecs = model(torch.tensor(padded_tokens).to(torch.int64), attention_mask=mask)
    # First vector is for [CLS] token, represents the whole sentence
    return word_vecs[0][:,0,:].numpy()

def evaluate(classifier, test_features, test_labels):
    return classifier.score(test_features, test_labels)

df = get_dataframe()

labels = get_labels(df)
tokenizer = get_tokenizer()
tokens = get_tokens(df, tokenizer)
padded = pad_tokens(tokens)
model = get_model()
vecs = get_bert_sentence_vectors(model, padded)

train_features, test_features, train_labels, test_labels = train_test_split(vecs, labels)

#train RandomForestClassifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

def train_RFC(train_features,train_labels):
  RFC = RandomForestClassifier()
  RFC.fit(train_features, train_labels)
  return RFC

def FiveXRFC(train_features, train_labels):
  RFC = RandomForestClassifier(n_estimators=500)
  RFC.fit(train_features, train_labels)
  return RFC

RFC = train_RFC(train_features, train_labels)
print(evaluate(RFC, test_features, test_labels))

FRFC = FiveXRFC(train_features, train_labels)
print(evaluate(FRFC, test_features, test_labels))

def get_tokens_generic(dataframe, tokenizer):
    return dataframe[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

def get_tokens_from_sentence(sentence):
  df = pd.DataFrame([[sentence]])
  return get_tokens_generic(df,get_tokenizer())

def get_bert_vecs_from_sentence(sentence):
  tokens = get_tokens_from_sentence(sentence)
  model = get_model()
  vecs =  get_bert_sentence_vectors(model, pad_tokens(tokens))
  return vecs

def predict_from_sentence(clf, sentence):
  vecs = get_bert_vecs_from_sentence(sentence)
  return clf.predict(vecs)

liberal = ["Trump and his supporters are the ones who turned politics into a weird cult.",
"The Republican Party is QAnon. MAGA is QAnon.",
"Tax cuts for the rich doesn't work. We've known that for decades. But they just keep chasing the same failed policies for no reason. This entire thing makes no sense.",
"What do you think would happen if Obama tried to pull out? The country would have collapsed to the Taliban, and the Republicans would have shredded Obama and accused him of failing their war.",
"It's just crazy that trump still has this much power over the party.  He lost them the House, he lost them the Senate and he lost them the White House.  Heck even when he 'won' he lost by 3 million votes.  Normally with that much failure a political party would not be willing to be held hostage yet instead they've gone to a full cult.",
"I wouldn’t call them “racial undertones”. The racism was very overt and clear cut to me.",
"Nope. The south is full of racists that live by a failed ideology of forced integration and the thoughts of losing their heritage causes them to embolden.",
"Yes, the police were at fault, thank you.",
"Well, it's not that simple. Many of Trump's policies, if you can call them that, don't really align with the GOP in general.",
"The majority is against Trump.",
"Raise the minimum wage."]

conservative = ["Harboring and assisting direct threats to the United States makes you a direct threat to the United States in my book.",
"Do you have proof that US soldiers are 'stealing Syrian oil'? That sounds like a populist lie you have been told.",
"Yup.  Trump actually wanted to reduce our dependence on China, unfortunately it's a lot longer process than a few years.  Biden wants us dependent on them again.",
"The Chinacrats from the Chinacratic Party are an open book...",
"'Rules for thee, but not for me!' -Democratic Party",
"Didn't WHO also say that the Floyd protests didn't present a risk for virus spreading but the stay at home order protests did? Anyone who doesn't realize they were being played about the seriousness of this whole thing needs to wake up.",
"Its the media. If you look up “media” in the dictionary youll see the definition on hypocrisy.",
"WHO and the NPR are hypocrites",
"This programming made possible by listeners like you...and the Chinese government.",
"Well in Biden's defense he probably won't remember having rallies."]

for sentence in liberal:
  print(sentence, predict_from_sentence(FRFC, sentence))
for sentence in conservative:
  print(sentence, predict_from_sentence(FRFC, sentence))
