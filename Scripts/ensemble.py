from os import name
import pandas as pd
import torch
import numpy as np
from collections import Counter

from utils import sorting_function, evaluate_ensemble, print_stats, load_prediction
from common import get_parser

parser = get_parser()
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

def max_vote():
    bert, xlnet, roberta, distilbert = load_prediction()

    target = []
    bert_pred = []
    xlnet_pred = []
    roberta_pred = []
    distilbert_pred = []

    for index in range(len(bert)):
       target.append(bert['target'][index])
       bert_pred.append(bert['y_pred'][index])
       xlnet_pred.append(xlnet['y_pred'][index])
       roberta_pred.append(roberta['y_pred'][index])
       distilbert_pred.append(distilbert['y_pred'][index])

    max_vote_df = pd.DataFrame()
    max_vote_df['target'] = target
    max_vote_df['bert'] = bert_pred
    max_vote_df['xlnet'] = xlnet_pred
    max_vote_df['roberta'] = roberta_pred
    max_vote_df['distilbert'] = distilbert_pred

    # print_stats(max_vote_df, bert, xlnet, roberta, distilbert)

    preds = []

    for index in range(len(max_vote_df)):
        values = max_vote_df.iloc[index].values[1:]
        sorted_values = sorted(Counter(values).items(), key = sorting_function, reverse=True)
        preds.append(sorted_values[0][0])
        
    max_vote_df['pred'] = preds

    evaluate_ensemble(max_vote_df)
    



if __name__=="__main__":
    max_vote()

