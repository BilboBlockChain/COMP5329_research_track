

#from torchtext.data import Field, TabularDataset, BucketIterator, Iterator




# preprocessing

# An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction dataset read from https://github.com/clinc/oos-eval/blob/master/data/data_full.json
import pandas as pd
import json

with open('data/data_full.json') as json_data:
    data = json.load(json_data)


train_df = pd.DataFrame(data['train'], columns=['X','y'])
val_df = pd.DataFrame(data['val'], columns=['X','y'])
test_df = pd.DataFrame(data['test'], columns=['X','y'])       #.append(pd.DataFrame(data['oos_test'], columns=['X','y'])) # we should append oos samples to test something to discuss, currently only in sample data

# create dictionary to transform string labels to int
label_dict = dict(zip(train_df['y'].unique(), range(len(train_df['y'].unique()))))

#training datasets
train_X = train_df['X'].to_list()
train_y = [label_dict[label] for label in train_df['y']]

val_X = val_df['X'].to_list()
val_y = [label_dict[label] for label in val_df['y']]

test_X = test_df['X'].to_list()
test_y = [label_dict[label] for label in test_df['y']]

# model
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn


class BERT_classifier():
    def __init__(self):
        super(BERT_classifier, self).__init__()
        self.bert = BertModel.from_pretrained(checkpoint) #currently froze BERT layers
        self.linear = nn.Linear(768, 256)
        self.output = nn.Linear(256, NUM_CLASS)

    def forward(self):

        # https://stackoverflow.com/questions/62705268/why-bert-transformer-uses-cls-token-for-classification-instead-of-average-over use the first token for classification see original BERT paper
        sequence_out, pooled_out = self.bert(inputs, attention_mask=mask)  # sequence_out.shape = (batch_size, sequence_length, 768)
        linear_out = self.linear(sequence_out[:, 0, :].view(-1, 768))  ## extract the 1st token's embeddings ala original paper, could consider averaging over states instead
        out = self.output(linear_out)

        return out


#hyper params

checkpoint = 'bert-base-uncased'
tokeniser = BertTokenizer.from_pretrained(checkpoint)

NUM_CLASS = len(label_dict)

num_epochs = 1
batch_size = 16
learning_rate = 0.01

criterion = nn.CrossEntropyLoss()

model = BERT_classifier() # You can pass the parameters if required to have more flexible model
model.to(torch.device("gpu")) ## can be gpu
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#test = tokeniser(train_X, padding = True, return_tensors="pt") #can explore tokenising before batch, outputs masked map and inputs tokens

#train
for epoch in range(num_epochs):
    train_loss = 0
    for ind in range(0, train_X.shape[0], batch_size):
        # batch implementation adopted from lab 4
        input_batch = train_X[ind:min(ind + batch_size, train_X.shape[0])]
        encoding_batch = tokeniser.batch_encode_plus(input_batch,max_length=128,padding=128, truncation=True) # run through max pad is still quite small seq length

        target_batch = train_y[ind:min(ind + batch_size, train_X.shape[0])]
        target_batch = torch.torch.from_numpy(target_batch).view(-1)

        optimizer.zero_grad()  # zero out gradient
        outputs = model.train(encoding_batch['input_ids'], encoding_batch['attention_mask'])  # bert takes in attention mask and ids
        loss = criterion(outputs, target_batch)  # evaluate loss for given epoch
        loss.backward()  # backprop
        optimizer.step()

        train_loss += loss.item()

    print(f'Epoch: {epoch + 1}, train loss: {train_loss}')








