import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import openml
import numpy as np
from datetime import datetime
import nltk
from nltk.corpus import words
import os
import seaborn as sns
from distilBert import CustomDataset, load_and_prepare_data
from tqdm import tqdm
import matplotlib.pyplot as plt
from bertviz import head_view, model_view


# Initialize tokenizer and model

def get_available_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Use Apple Metal if available
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

device = get_available_device()



def compute_feature_attention(input_text, num_features):
    """
    Compute attention for each feature in a single input text.
    """
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(input_text, return_tensors='pt').to(device)
        outputs = model(**inputs)

        # Extract attention tensors (shape: num_layers, batch_size, num_heads, seq_length, seq_length)
        attentions = outputs.attentions

        # Assume we're interested in the last layer's attention
        last_layer_attention = attentions[-1].squeeze(0)  # Remove batch dimension, shape: num_heads, seq_length, seq_length

        # average over all attention layers
        # t = torch.stack(attentions, dim=0)
        # t = t.squeeze(1).mean(dim=0).mean(dim=0)  # avg over all attention layers
        # attention_avg_heads = t

        # Average over heads
        attention_avg_heads = last_layer_attention.mean(dim=0)  # Shape: seq_length, seq_length

        # check attention values wrt cls and sep
        cls_given = attention_avg_heads[:, 0]
        cls_gave = attention_avg_heads[0, :]
        sep_given = attention_avg_heads[:, -1]
        sep_gave = attention_avg_heads[-1, :]
        sep_given = sep_gave

        # # Tokenize input to find commas
        # tokens = tokenizer.tokenize(input_text)

        # # create heatmaps
        # toks = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(attention_avg_heads.cpu(), xticklabels=toks, yticklabels=toks, cmap='viridis')
        # plt.show()
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(attention_avg_heads[1:-1, 1:-1].cpu(), xticklabels=toks[1:-1], yticklabels=toks[1:-1], cmap='viridis')
        # plt.show()

        # transpose attention matrix
        # attention_avg_heads = attention_avg_heads.T

        # Identify indices of commas and segments
        comma_indices = [i + 1 for i, token in enumerate(tokens) if token == ',']
        segment_indices = [1] + [index + 1 for index in comma_indices] + [len(tokens) + 1]  # Start of each feature, +1 to skip the comma itself
        if len(comma_indices) < num_features - 1:  # Ensure there are n features
            segment_indices.append(len(tokens) + 1)



        # Sum attention values for each feature
        feature_attentions = []
        feature_attentions_sep_given = []
        for start, end in zip(segment_indices[:-1], segment_indices[1:]):
            feature_tokens = end - start
            feature_attention = attention_avg_heads[:, start:end].sum().item() / feature_tokens
            feature_attention_sep_given = sep_given[start:end].sum().item() / feature_tokens

            # feature_attention = feature_attention / feature_tokens
            feature_attentions.append(feature_attention)
            feature_attentions_sep_given.append(feature_attention_sep_given)

        return feature_attentions, feature_attentions_sep_given


# Dataset of inputs
nonsense = None
samples, labels = load_and_prepare_data('iris', nonsense)
model_path = "./models/iris"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()
X_train, X_val, y_train, y_val = train_test_split(samples, labels, test_size=0.2, random_state=42)

# Compute attention for each feature across all inputs
num_features = 1
for letter in list(X_train[0]):
    if letter == ',':
        num_features += 1

if True:
    inputs = tokenizer(X_val[len(X_val) // 2], return_tensors='pt').to(device)
    input_ids = inputs['input_ids']
    # token_type_ids = inputs['token_type_ids']
    outputs = model(input_ids)
    attentions = outputs.attentions
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    model_view(attentions, tokens)


dataset = X_val
feature_attention_sums = np.zeros(shape=num_features)
feature_attention_sep_given_sums = np.zeros(shape=num_features)
for input_text in tqdm(dataset):
    feature_attentions, feature_attention_sep_given = compute_feature_attention(input_text, num_features)
    # feature_attention_sums = [sum(x) for x in zip(feature_attention_sums, feature_attentions)]
    feature_attention_sums += feature_attentions
    feature_attention_sep_given_sums += feature_attention_sep_given

# Compute average attention per feature
# num_samples = len(dataset)
average_feature_attentions = feature_attention_sums / sum(feature_attention_sums)
average_feature_attentions_sep_given = feature_attention_sep_given_sums / sum(feature_attention_sep_given_sums)

print("Average Attention per Feature:", average_feature_attentions.tolist())
print("Average Attention SEP Given:", average_feature_attentions_sep_given.tolist())
