from pre_processing import preprocess
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
from utils import eval 
from bert_model import BERT_classifier
from transformers import BertTokenizer,RobertaTokenizer
#params
NUM_CLASS = 151
num_epochs = 5
batch_size = 16
learning_rate = 5e-5
momentum = 0.6

#get batches
dataset, checkpoint = preprocess(tokenizer_type = RobertaTokenizer, checkpoint = 'bert-base-uncased') #pull bert tokenised data and checkpoint from preprocess


train_dataloader = DataLoader(dataset['train'], shuffle=True, batch_size=batch_size)
val_dataloader = DataLoader(dataset['val'], batch_size=batch_size)
test_dataloader = DataLoader(dataset['test'], batch_size=batch_size)


test = next(iter(train_dataloader))
test

# get model

criterion = nn.CrossEntropyLoss()
device = torch.device("cuda")

model = BERT_classifier(checkpoint, NUM_CLASS) # uses same checkpoint as bert tokenizer - dont worry about the warning, it is expected while we are extracting only the CLS token
model.to(device)


#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

#train

from tqdm.auto import tqdm
num_training_steps = num_epochs * len(train_dataloader)


for epoch in range(num_epochs):
    train_loss = 0
    with tqdm(train_dataloader, unit="batch") as tepoch:
        model.train()

        for batch in tepoch:
    #for batch in train_dataloader:
            tepoch.set_description(f"Epoch {epoch}")
            inputs = torch.stack(batch['input_ids'][0], dim = 1).to(device) #convert list of tensors to tensors
            targets = batch['y'].to(device)
            mask = torch.stack(batch['attention_mask'][0], dim = 1).to(device)

            outputs = model(inputs, mask)

            optimizer.zero_grad()  # zero out gradient


            loss = criterion(outputs, targets)  # evaluate loss for given epoch
            loss.backward()  # backprop
            optimizer.step()

            #acc
            batch_perf = np.array(outputs.detach().cpu().numpy()).argmax(axis=1) == targets.detach().cpu().numpy()
            batch_acc = np.sum(np.array(batch_perf)/len(batch_perf))


            train_loss += loss.item()
            #progress_bar.update(1)
            tepoch.set_postfix(loss=loss.item(), batch_acc= batch_acc  )

    model.eval()




    eval(val_dataloader)
    eval(test_dataloader)
    eval(train_dataloader)












