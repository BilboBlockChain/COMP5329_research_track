from pre_processing import preprocess
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np

#params
NUM_CLASS = 151
num_epochs = 1
batch_size = 16
learning_rate = 0.01

#get batches
dataset, checkpoint = preprocess() #pull bert tokenised data and checkpoint from preprocess

train_dataloader = DataLoader(dataset['train'], shuffle=True, batch_size=batch_size)
val_dataloader = DataLoader(dataset['val'], batch_size=batch_size)
test_dataloader = DataLoader(dataset['test'], batch_size=batch_size)

# get model
from bert_model import BERT_classifier

criterion = nn.CrossEntropyLoss()
device = torch.device("cuda")

model = BERT_classifier(checkpoint, NUM_CLASS) # uses same checkpoint as bert tokenizer - dont worry about the warning, it is expected while we are extracting only the CLS token
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#test = tokeniser(train_X, padding = True, return_tensors="pt") #can explore tokenising before batch, outputs masked map and inputs tokens

#inputs = next(iter(train_dataloader))
#inputs.items()


#train

from tqdm.auto import tqdm
num_training_steps = num_epochs * len(train_dataloader)
progress_bar = tqdm(range(num_training_steps))



for epoch in range(num_epochs):
    train_loss = 0
    for batch in train_dataloader:
        # batch implementation adopted from lab 4
        inputs = torch.stack(batch['input_ids'][0]).to(device) #convert list of tensors to tensors
        targets = batch['y'].to(device)
        mask = torch.stack(batch['attention_mask'][0]).to(device)

        outputs = model(inputs, mask)

        optimizer.zero_grad()  # zero out gradient

        #outputs = model.train(encoding_batch['input_ids'], encoding_batch['attention_mask'])  # bert takes in attention mask and ids
        loss = criterion(outputs, targets)  # evaluate loss for given epoch
        loss.backward()  # backprop
        optimizer.step()

        train_loss += loss.item()
        progress_bar.update(1)

    print(f'Epoch: {epoch + 1}, train loss: {train_loss}')




