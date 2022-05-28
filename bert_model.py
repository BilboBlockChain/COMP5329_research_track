# model
from transformers import  BertModel
import torch
import torch.nn as nn
import torch.nn.functional as f


class BERT_classifier(nn.Module): #inherit from pytorch module
    def __init__(self, checkpoint, num_class):
        super(BERT_classifier, self).__init__()
        self.bert = BertModel.from_pretrained(checkpoint)
        self.head = nn.Linear(768, 256)
        self.output = nn.Linear(256, num_class)

    def forward(self, inputs, mask=None):

        # https://stackoverflow.com/questions/62705268/why-bert-transformer-uses-cls-token-for-classification-instead-of-average-over use the first token for classification see original BERT paper
        #bert_out = self.bert(inputs, attention_mask=mask)  # out['last_hidden_state'].shape = (batch_size,sequence_length, 768)
        bert_out = self.bert(inputs)
                               # out['last_hidden_state'].shape = (batch_size,sequence_length, 768)
        #print(bert_out[1].shape, bert_out[0].shape )
        #head_out = f.relu(self.head(bert_out[1]))

        #print(head_out.shape)
        head_out =self.head(bert_out['last_hidden_state'][:, 0, :].view(-1, 768)) ## extract the 1st token's embeddings ala original paper, could consider averaging over states instead
        out = self.output(head_out)
        #print(out.shape)
        return out