import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel

from common import get_parser

parser = get_parser()
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

class BertFGBC(nn.Module):
    def __init__(self):
        super().__init__()
        self.Bert = BertModel.from_pretrained(args.pretrained_model_name)
        self.Bert_drop = nn.Dropout(args.dropout)
        self.out = nn.Linear(args.bert_hidden, args.classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _,o2 = self.Bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )

        bo = self.Bert_drop(o2)
        output = self.out(bo)

        return output