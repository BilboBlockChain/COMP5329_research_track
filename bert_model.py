# model
from transformers import  BertModel
import torch
import torch.nn as nn


class BERT_classifier(nn.Module): #inherit from pytorch module
    def __init__(self, checkpoint, num_class):
        super(BERT_classifier, self).__init__()
        self.bert = BertModel.from_pretrained(checkpoint)
        self.relu = nn.ReLU(nn.Linear(768, 256))
        self.output = nn.Linear(256, num_class)

    def forward(self, inputs, mask):

        # https://stackoverflow.com/questions/62705268/why-bert-transformer-uses-cls-token-for-classification-instead-of-average-over use the first token for classification see original BERT paper
        sequence_out, pooled_out = self.bert(inputs, attention_mask=mask)  # sequence_out.shape = (batch_size, sequence_length, 768)
        relu_out = self.relu(sequence_out[:, 0, :].view(-1, 768))  ## extract the 1st token's embeddings ala original paper, could consider averaging over states instead
        out = self.output(relu_out)
        return out