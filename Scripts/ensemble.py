from os import name
import pandas as pd
import torch
import numpy as np
from collections import Counter

from evaluate import test_evaluate
from engine import test_eval_fn_ensemble
from utils import sorting_function, evaluate_ensemble, print_stats, load_prediction, set_device, load_models, generate_dataset_for_ensembling
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

def averaging():
    bert, xlnet, roberta, distilbert = load_models()
    test_df = pd.read_csv(f'{args.dataset_path}test.csv').dropna()
    device = set_device()

    bert.to(device)
    test_data_loader = generate_dataset_for_ensembling(pretrained_model="bert-base-uncased", df =test_df)
    test_eval_fn_ensemble(test_df, test_data_loader, bert, device, pretrained_model="bert-base-uncased")
    del bert, test_data_loader

    # xlnet.to(device)
    # test_data_loader = generate_dataset_for_ensembling(pretrained_model="xlnet-base-cased", df=test_df)
    # test_evaluate(test_df, test_data_loader, bert, device, pretrained_model="xlnet-base-cased")
    # del xlnet, test_data_loader

    # roberta.to(device)
    # test_data_loader = generate_dataset_for_ensembling(pretrained_model="roberta-base", df=test_df)
    # test_evaluate(test_df, test_data_loader, bert, device, pretrained_model="roberta-base")
    # del roberta, test_data_loader

    # distilbert.to(device)
    # test_data_loader = generate_dataset_for_ensembling(pretrained_model="distilbert-base-uncased", df=test_df)
    # test_evaluate(test_df, test_data_loader, bert, device, pretrained_model="distilbert-base-uncased")
    # del distilbert, test_data_loader






if __name__=="__main__":
    max_vote()
    # averaging()

