import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel, RobertaModel, XLNetModel, DistilBertModel

from common import get_parser

parser = get_parser()
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

class BertFGBC(nn.Module):
    def __init__(self):
        super().__init__()
        self.Bert = BertModel.from_pretrained(args.pretrained_model)
        self.Bert_drop = nn.Dropout(args.dropout)
        self.out = nn.Linear(args.bert_hidden, args.classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _,last_hidden_state = self.Bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )
        print(f'{last_hidden_state.shape}-"Last Hidden State\n"{last_hidden_state}')

        bo = self.Bert_drop(last_hidden_state)
        output = self.out(bo)

        return output

class RobertaFGBC(nn.Module):
    def __init__(self):
        super().__init__()
        self.Roberta = RobertaModel.from_pretrained(args.pretrained_model)
        self.Roberta_drop = nn.Dropout(args.dropout)
        self.out = nn.Linear(args.roberta_hidden, args.classes)

    def forward(self, input_ids, attention_mask):
        _,last_hidden_state = self.Roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        bo = self.Roberta_drop(last_hidden_state)
        output = self.out(bo)

        return output

class DistilBertFGBC(nn.Module):
    def __init__(self):
        super().__init__()
        self.DistilBert = DistilBertModel.from_pretrained(args.pretrained_model)
        self.DistilBert_drop = nn.Dropout(args.dropout)
        self.out = nn.Linear(args.distilbert_hidden, args.classes)

    def forward(self, input_ids, attention_mask):
        last_hidden_state = self.DistilBert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        mean_last_hidden_state = self.pool_hidden_state(last_hidden_state)
        
        bo = self.DistilBert_drop(mean_last_hidden_state)
        output = self.out(bo)

        return output

    def pool_hidden_state(self, last_hidden_state):
        last_hidden_state = last_hidden_state[0]
        mean_last_hidden_state = torch.mean(last_hidden_state, 1)
        return mean_last_hidden_state

class XLNetFGBC(nn.Module):
    def __init__(self):
        super().__init__()
        self.XLNet = XLNetModel.from_pretrained(args.pretrained_model)
        self.XLNet_drop = nn.Dropout(args.dropout)
        self.out = nn.Linear(args.xlnet_hidden, args.classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        last_hidden_state = self.XLNet(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )
        mean_last_hidden_state = self.pool_hidden_state(last_hidden_state)

        bo = self.XLNet_drop(mean_last_hidden_state)
        output = self.out(bo)

        return output
        
    def pool_hidden_state(self, last_hidden_state):
        last_hidden_state = last_hidden_state[0]
        mean_last_hidden_state = torch.mean(last_hidden_state, 1)
        return mean_last_hidden_state