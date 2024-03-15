import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import TrainerCallback
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import openml
import numpy as np
import os
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import KFold
from datetime import datetime


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_available_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Use Apple Metal if available
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

device = get_available_device()

def load_and_prepare_data(dataset_name):
    if dataset_name == 'iris':
        from sklearn.datasets import load_iris
        data = load_iris()
        X, y = data.data, data.target

        # Convert features to text for iris dataset
        def convert_to_text(data):
            text_data = []
            for sample in data:
                description = f"The flower has a sepal length of {sample[0]} cm, a sepal width of {sample[1]} cm, a petal length of {sample[2]} cm, and a petal width of {sample[3]} cm."
                text_data.append(description)
            return text_data

        X_text = convert_to_text(X)
    elif dataset_name == 'LED':
        dataset = openml.datasets.get_dataset(40496)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        # Assuming X is already in a suitable format (e.g., images flattened into vectors)
        # Convert digits to text description (simplistic approach)
        y = [int(val) - 1 for val in y.array]

        # X = [true_digits_labels[label] for label in y]
        def convert_to_text(data):
            # return [f"The boolean state of the LED is: {str(sample)}" for sample in data]
            return [f"{str(sample)}" for sample in data]

        X_text = convert_to_text(X.to_numpy())  # for X is df
        # X_text = convert_to_text(X)
    elif dataset_name == 'OPT':
        dataset = openml.datasets.get_dataset(28)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        y = [int(val) for val in y.array]

        def convert_to_text(data):
            return [f"{np.array2string(sample, separator=', ', max_line_width=np.inf)}" for sample in data]

        X_text = convert_to_text(X.to_numpy())  # for X is df

    elif dataset_name == 'adult':
        dataset = openml.datasets.get_dataset(1590)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        y = [0 if val == '<=50K' else 1 for val in y.array]

        def convert_to_text(data):
            return [f"{np.array2string(sample, separator=', ', max_line_width=np.inf)}" for sample in data]

        X_text = convert_to_text(X.to_numpy())  # for X is df

    if X_text != None:
        # Shuffle the dataset
        _X_text_shuffled, _y_shuffled = shuffle(X_text, y, random_state=42)
        return _X_text_shuffled, _y_shuffled


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx], device=device) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], device=device)
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions[0].argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}


class EpochLoggingCallback(TrainerCallback):
    def __init__(self, filename='train_log.txt'):
        self.file_name = filename
        # Ensure the file is cleared at the beginning of training
        with open(self.file_name, 'a') as file:
            file.write("Training Log\n")

    def on_epoch_end(self, args, state, control, **kwargs):
        with open(self.file_name, 'a') as file:
            file.write("--------------------------------\n")
            file.write(f"\nEpoch: {state.epoch}\n")
            train_results = trainer.predict(train_dataset)
            file.write("Trianing Accuracy: " + str(train_results.metrics['test_accuracy']) + " \n")
            test_results = trainer.predict(val_dataset)
            file.write("Test Accuracy: " + str(test_results.metrics['test_accuracy']) + " \n")
            file.write("--------------------------------\n")



# X_train, X_val, y_train, y_val = train_test_split(X_text_shuffled, y_shuffled, test_size=0.2, random_state=42)
# train_encodings = tokenizer(X_train, truncation=True, padding=True)
# val_encodings = tokenizer(X_val, truncation=True, padding=True)
# train_dataset = CustomDataset(train_encodings, y_train)
# val_dataset = CustomDataset(val_encodings, y_val)
# model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(set(y_shuffled)))
# total_train_examples = len(train_dataset)
def train_test(dataset_name):
    global train_dataset, val_dataset, trainer
    filename = dataset_name + "_kf_train_log.txt"
    # Load the dataset (either 'iris' or 'digits')
    X_text_shuffled, y_shuffled = load_and_prepare_data(dataset_name)
    # Tokenize the text descriptions
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    encodings = tokenizer(X_text_shuffled, truncation=True, padding=True)
    dataset = CustomDataset(encodings, y_shuffled)
    # dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], torch.tensor(labels))
    total_train_examples = int(len(dataset) * 0.8)
    batch_size = 16
    steps_per_epoch = total_train_examples // batch_size
    logging_and_eval_steps = steps_per_epoch * 20  # Log every 10 epochs
    k = 5  # Number of folds
    kf = KFold(n_splits=k)
    total_accuracy = 0
    with open(filename, 'w') as file:
        file.write(f"Started Training\n")
        file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write("--------------------------------\n")
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        with open(filename, 'a') as file:
            file.write(f"FOLD {fold}\n")
            file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write("--------------------------------\n")

        # Split the data

        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)

        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', output_attentions=True,
                                                                    num_labels=len(set(y_shuffled)))



        # Add your training loop here
        # Train your model on the training set and evaluate it on the validation set
        training_args = TrainingArguments(
            output_dir='./results',
            save_strategy='epoch',
            save_total_limit=3,
            num_train_epochs=20,
            per_device_train_batch_size=batch_size,
            # per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            # logging_dir='./logs',
            # logging_strategy="steps",  # Log based on steps
            # logging_steps=logging_and_eval_steps,
            # evaluation_strategy="steps",
            # eval_steps=logging_and_eval_steps,
            # logging_strategy="epoch",         # Adjusted to log based on epochs
            # evaluation_strategy="epoch",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EpochLoggingCallback(filename)],  # Added custom callback

        )

        trainer.train()

        acc = trainer.predict(val_dataset).metrics['test_accuracy']
        total_accuracy += acc
    with open(filename, 'a') as file:
        file.write("--------------------------------\n")
        file.write("--------------------------------\n")
        file.write(f'total accuracy: {total_accuracy / k}\n')
    # print(f'total accuracy: {total_accuracy / k}\n')



if __name__ == '__main__':
    datasets_names = ['iris']
    for dataset in datasets_names:
        train_test(dataset)