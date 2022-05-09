

# preprocessing

# An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction dataset read from https://github.com/clinc/oos-eval/blob/master/data/data_full.json
from datasets import Dataset, DatasetDict
from transformers import BertTokenizer
import pandas as pd
import json
from torch.utils.data import DataLoader

def preprocess(): #currently doesnt take params but we probably want to  include some if we decide to make changes to preprocessing for testing results

    #open json in pandas for easy appending and one hot encode
    with open('data/data_full.json') as json_data:
        data = json.load(json_data)

    train_df = pd.DataFrame(data['train'], columns=['text','label'])
    val_df = pd.DataFrame(data['val'], columns=['text','label'])
    test_df = pd.DataFrame(data['test'], columns=['text','label']).append(pd.DataFrame(data['oos_test'], columns=['text','label']),ignore_index=True ) # we should append oos samples for test not training sets

    #convert label to int
    label_dict = dict(zip(train_df['label'].unique(), range(len(train_df['label'].unique()))))
    label_dict['oos'] = max(label_dict.values()) + 1 #add out of sample label to dictionary though no training data contains it

    train_df['y'] = [label_dict[label] for label in train_df['label']]
    val_df['y'] = [label_dict[label] for label in val_df['label']]
    test_df['y'] = [label_dict[label] for label in test_df['label']]

#create dataset for bulk BERT tokenization

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    ds = DatasetDict()

    ds['train'] = train_dataset
    ds['val'] = val_dataset
    ds['test'] = test_dataset

    checkpoint = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(checkpoint)

    def tokenize_function(data):

        tokenized_batch = tokenizer(data["text"], padding="max_length", truncation=True, return_tensors='pt')
        return tokenized_batch


    tokenized_datasets = ds.map(tokenize_function)

    return tokenized_datasets, checkpoint

if __name__ == "__main__":
    dataset, checkpoint = preprocess()






#MANUAL DEPRECATED
# create dictionary to transform string labels to int
#label_dict = dict(zip(train_df['y'].unique(), range(len(train_df['y'].unique()))))

#training datasets
#train_X = train_df['X'].to_list()
#train_y = [label_dict[label] for label in train_df['y']]

#val_X = val_df['X'].to_list()
#val_y = [label_dict[label] for label in val_df['y']]

#test_X = test_df['X'].to_list()
#test_y = [label_dict[label] for label in test_df['y']]





#from transformers import BertTokenizer

#checkpoint = 'bert-base-uncased'
#tokeniser = BertTokenizer.from_pretrained(checkpoint)

#tokeniser.batch_encode_plus(input_batch,max_length=128,padding=128, truncation=True)


