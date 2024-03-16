import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import TrainerCallback
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import openml
import numpy as np
from datetime import datetime
import nltk
from nltk.corpus import words
import os

prng = np.random.RandomState(42)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_available_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Use Apple Metal if available
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

device = get_available_device()


# todo change dataset to python
def add_nonsense_float_features(dataset, num_features=1):
    dim = len(dataset)
    for i in range(num_features):
        random_col = 100 * prng.standard_normal(size=dim)
        # random_col = np.random.normal(loc=0, scale=100, size=dim, dtype=np.float32)
        random_col = random_col.reshape((-1, 1))
        random_col = random_col.astype(np.float32).round(2)
        dataset = np.concatenate((dataset, random_col), axis=1)
    return dataset

def add_nonsense_int_features(dataset, num_features=1):
    dim = len(dataset)
    for i in range(num_features):
        random_col = np.random.randint(low=-dim, high=dim, size=dim, dtype=np.int32)
        random_col = random_col.reshape((-1, 1))
        random_col = random_col.astype(np.float32).round(3)
        dataset = np.concatenate((dataset, random_col), axis=1)
    return dataset

def add_nonsense_bool_features(dataset, num_features=1):
    dim = len(dataset)
    for i in range(num_features):
        random_col = np.random.choice([True, False], size=dim)
        random_col = random_col.reshape((-1, 1))
        dataset = np.concatenate((dataset, random_col), axis=1)
    return dataset

def add_nonsense_string_features(dataset, num_features=1):
    dim = len(dataset)
    chars = np.array(list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))
    for i in range(num_features):
        lengths = np.random.randint(2, 8, size=dim)
        random_col = np.array([''.join(np.random.choice(chars, length)) for length in lengths])
        random_col = random_col.reshape((-1, 1))
        dataset = np.concatenate((dataset, random_col), axis=1)
    return dataset

def add_nonsense_word_features(dataset, num_features=1):
    dim = len(dataset)
    nltk.download('words')
    english_words = np.array(words.words())
    for i in range(num_features):
        indices = np.random.randint(0, len(english_words), size=dim)
        random_col = english_words[indices]
        random_col = random_col.reshape((-1, 1))
        dataset = np.concatenate((dataset, random_col), axis=1)
    return dataset


def convert_to_text(data):
    return [f"{np.array2string(sample, separator=', ', max_line_width=np.inf)[1:-1]}" for sample in data]


def load_and_prepare_data(dataset_name, nonesense_features: list = None, return_numbers=False):
    X = None
    y = None
    if dataset_name == 'iris':
        from sklearn.datasets import load_iris
        data = load_iris()
        X, y = data.data.astype(np.float32), data.target
        # Convert features to text for iris dataset
        # def convert_to_text(data):
        #     text_data = []
        #     for sample in data:
        #         description = str(sample)[1:-1]
        #         # description = f"The flower has a sepal length of {sample[0]} cm, a sepal width of {sample[1]} cm, a petal length of {sample[2]} cm, and a petal width of {sample[3]} cm."
        #         text_data.append(description)
        #     return text_data
        # X_text = convert_to_text(X)
    elif dataset_name == 'LED':
        dataset = openml.datasets.get_dataset(40496)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        # Assuming X is already in a suitable format (e.g., images flattened into vectors)
        # Convert digits to text description (simplistic approach)
        y = [int(val) - 1 for val in y.array]
        X = X.to_numpy()
        # X = [true_digits_labels[label] for label in y]
        # def convert_to_text(data):
        #     # return [f"The boolean state of the LED is: {str(sample)}" for sample in data]
        #     return [f"{str(sample)[1:-1]}" for sample in data]
        # X_text = convert_to_text(X.to_numpy()) #for X is df
        # X_text = convert_to_text(X)
    elif dataset_name == 'OPT':
        dataset = openml.datasets.get_dataset(28)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        y = [int(val) for val in y.array]
        X = X.to_numpy()
        # def convert_to_text(data):
        #     return [f"{np.array2string(sample, separator=', ', max_line_width=np.inf)[1:-1]}" for sample in data]
        # X_text = convert_to_text(X.to_numpy()) #for X is df

    elif dataset_name == 'adult':
        dataset = openml.datasets.get_dataset(1590)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        y = [0 if val=='<=50K' else 1 for val in y.array]
        if return_numbers:
            le = LabelEncoder()
            for col in X.columns:
                if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                    X[col] = le.fit_transform(X[col])

        X = X.to_numpy()
        # def convert_to_text(data):
        #     return [f"{np.array2string(sample, separator=', ', max_line_width=np.inf)[1:-1]}" for sample in data]
        # X_text = convert_to_text(X.to_numpy())  # for X is df

    if nonesense_features is not None and X is not None:
        for feature in nonesense_features:
            if feature == 'int':
                X = add_nonsense_int_features(X)
            elif feature == 'float':
                X = add_nonsense_float_features(X)
            elif feature == 'bool':
                X = add_nonsense_bool_features(X)
            elif feature == 'string':
                X = add_nonsense_string_features(X)
            elif feature == 'words':
                X = add_nonsense_word_features(X)

    prng.shuffle(X.T)

    if return_numbers:
        _X_text_shuffled, _y_shuffled = shuffle(X, y, random_state=42)
        return _X_text_shuffled, _y_shuffled

    if X is not None:
        # Shuffle the dataset
        X_text = convert_to_text(X)
        print(X_text[0])
        _X_text_shuffled, _y_shuffled = shuffle(X_text, y, random_state=42)
        return _X_text_shuffled, _y_shuffled



class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions[0].argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

def output_file_name(dataset_name, nonsense_features: list = None, folds: int = 1):
    if nonsense_features is None:
        nonsense_features = ''
    else:
        nonsense_features = '_' + str(nonsense_features)
    kfold_addition = '_' + str(folds) + 'f' if folds > 1 else ''
    ending = '_train_log.txt'
    file_name = dataset_name + nonsense_features + kfold_addition + ending
    return file_name

class EpochLoggingCallback(TrainerCallback):
    def __init__(self, data_name="", nonsense: list = None):
        self.file_name = data_name + '_train_log.txt'
        if nonsense is not None:
            self.file_name = data_name + "_" + str(nonsense) + '_train_log.txt'
        # Ensure the file is cleared at the beginning of training
        with open(self.file_name, 'w') as file:
            file.write("Training Log\n")
            file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    def on_epoch_end(self, args, state, control, **kwargs):
        with open(self.file_name, 'a') as file:
            file.write(f"\nEpoch: {state.epoch}\n")
            train_results = trainer.predict(train_dataset)
            file.write("Trianing Accuracy: "+ str(train_results.metrics['test_accuracy']) + " \n")
            test_results = trainer.predict(val_dataset)
            file.write("Test Accuracy: " + str(test_results.metrics['test_accuracy']) + " \n")

            # # The Trainer stores current training and validation metrics in `state.log_history`
            # train_logs = [log for log in state.log_history if "loss" in log]
            # eval_logs = [log for log in state.log_history if "eval_loss" in log]
            # if train_logs:
            #     print(f"Training Loss: {train_logs[-1]['loss']}")
            # if eval_logs:
            #     print(f"Validation Accuracy: {eval_logs[-1].get('eval_accuracy', 'N/A')}")




def train_test_distilbert(dataset_name, nonsense_features: list = None, folds: int = 1):
    global train_dataset, val_dataset, trainer
    filename = output_file_name(dataset_name, nonsense_features, folds)
    # load dataset
    X_text_shuffled, y_shuffled = load_and_prepare_data(dataset_name, nonsense_features)
    # Tokenize the text descriptions1
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    X_train, X_val, y_train, y_val = train_test_split(X_text_shuffled, y_shuffled, test_size=0.2, random_state=42)
    train_encodings = tokenizer(X_train, truncation=True, padding=True, return_tensors="pt")
    val_encodings = tokenizer(X_val, truncation=True, padding=True, return_tensors="pt")
    train_dataset = CustomDataset(train_encodings, y_train)
    val_dataset = CustomDataset(val_encodings, y_val)
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', output_attentions=True,
                                                                num_labels=len(set(y_shuffled)))
    total_train_examples = len(train_dataset)
    batch_size = 16
    steps_per_epoch = total_train_examples // batch_size
    logging_and_eval_steps = steps_per_epoch * 20  # Log every 10 epochs
    training_args = TrainingArguments(
        output_dir='./results',
        save_strategy='epoch',
        save_total_limit=3,
        num_train_epochs=17,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_strategy="steps",  # Log based on steps
        logging_steps=logging_and_eval_steps,
        evaluation_strategy="steps",
        eval_steps=logging_and_eval_steps,
        # logging_strategy="epoch",         # Adjusted to log based on epochs
        # evaluation_strategy="epoch",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EpochLoggingCallback(dataset_name, nonsense_features)],  # Added custom callback
    )
    # with open('training_log.txt', 'a') as file:
    #     train_results = trainer.predict(train_dataset)
    #     file.write("Trianing Accuracy: " + str(train_results.metrics['test_accuracy']) + " \n")
    #     test_results = trainer.predict(val_dataset)
    #     file.write("Test Accuracy: " + str(test_results.metrics['test_accuracy']) + " \n")
    #
    # exit()
    trainer.train()
    nonsense_path = str(nonsense_features)[1:-1]
    if nonsense_path != '' and nonsense_features is not None:
        dataset_name = "_" + nonsense_path
    model_path = "./models/" + dataset_name
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

if __name__ == '__main__':
    dataset_name = "iris"
    train_test_distilbert(dataset_name)
# train_results = trainer.predict(train_dataset)
# test_results = trainer.evaluate()
# print(train_results.metrics['test_accuracy'])
# # print(test_results)
# trainer.train()
# train_results = trainer.predict(train_dataset)
# print(train_results.metrics['test_accuracy'])
# trainer.train()
# train_results = trainer.predict(train_dataset)
# print(train_results.metrics['test_accuracy'])
# print(trainer.predict(val_dataset).metrics['test_accuracy'])

