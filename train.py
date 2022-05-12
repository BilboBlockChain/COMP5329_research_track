from pre_processing import preprocess
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
import torchmetrics as tm
from laplace import Laplace

import pandas as pd
#params
NUM_CLASS = 151
num_epochs = 1
batch_size = 16
learning_rate = 5e-5
momentum = 0.6

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


    def eval(dataload):
        out = []
        targ = []
        for batch in tqdm(dataload):
            inputs = torch.stack(batch['input_ids'][0], dim=1).to(device)  # convert list of tensors to tensors
            targets = batch['y'].to(device)
            mask = torch.stack(batch['attention_mask'][0], dim=1).to(device)
            with torch.no_grad():
                outputs = model(inputs, mask)
            out.append(outputs)
            out_torch = torch.cat(out).squeeze()
            pred_torch = torch.argmax(out_torch, dim=-1)
            targ.append(targets)
            target_torch = torch.cat(targ).squeeze()

        return torch.sum(pred_torch == target_torch).detach().cpu().numpy()/len(torch.cat(out).squeeze()), out_torch, target_torch


    val, val_out_torch, val_target_torch = eval(val_dataloader)
    train, train_out_torch, train_target_torch = eval(train_dataloader)

    CE = tm.CalibrationError().to(device)

    print(f"Train Acc: {train}, Train Calibration Error {CE(train_out_torch,train_target_torch)}|Val Acc:{val}, Val Calibration Error {CE(val_out_torch,val_target_torch)}")



#test results

    test, test_out_torch, test_target_torch = eval(test_dataloader)
    #test = eval(test_dataloader)

    CE(test_out_torch,test_target_torch)
    CE(test_out_torch[4500:,:],test_target_torch[4500:])



# laplaceAlt approx




X = torch.from_numpy(np.vstack(np.concatenate(train_dataloader.dataset.to_pandas()['input_ids'].values, axis=0 )))
y = torch.from_numpy(train_dataloader.dataset.to_pandas()['y'].values)


class LaplaceDataset(Dataset):
    def __init__(self, X, y):
        super(Dataset, self).__init__()
        self.X = X
        self.y = y
    def __len__(self):
        return len(X)
    def __getitem__(self):
        return X,y


  trainLaplace =  LaplaceDataset(X = X, y = y)

  trainLaplace


  train_lap_dataloader = DataLoader(trainLaplace,shuffle=True, batch_size=batch_size)


  la = Laplace(model, 'classification',
               subset_of_weights='all',
               hessian_structure='diag')
  la.fit(train_lap_dataloader)


  next(iter(train_lap_dataloader.dataset))