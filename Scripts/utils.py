import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, f1_score, accuracy_score, precision_score, recall_score
from common import get_parser

parser = get_parser()
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_validate_test_split(df, train_percent=0.6, validate_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test

def sorting_function(val):
    return val[1]    

def load_prediction():
    bert_path = (f'{args.output_path}bert-base-uncased---test_acc---0.8153233413688292.csv')
    xlnet_path = (f'{args.output_path}xlnet-base-cased---test_acc---0.8411068022219893.csv')
    roberta_path = (f'{args.output_path}roberta-base---test_acc---0.8470810187611362.csv')
    distilbert_path = (f'{args.output_path}distilbert-base-uncased---test_acc---0.820039828110261.csv')

    bert = pd.read_csv(bert_path)
    xlnet = pd.read_csv(xlnet_path)
    roberta = pd.read_csv(roberta_path)
    distilbert = pd.read_csv(distilbert_path)

    return bert, xlnet, roberta, distilbert

def print_stats(max_vote_df, bert, xlnet, roberta, distilbert):
    print(max_vote_df.head())
    print(f'---Ground Truth---\n{bert.target.value_counts()}')
    print(f'---Bert---\n{bert.y_pred.value_counts()}')
    print(f'---XLNet---\n{xlnet.y_pred.value_counts()}')
    print(f'---Roberta---\n{roberta.y_pred.value_counts()}')
    print(f'---DistilBert---\n{distilbert.y_pred.value_counts()}')

def evaluate_ensemble(max_vote_df):
    y_test = max_vote_df['target'].values
    y_pred = max_vote_df['pred'].values
    acc = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print('Accuracy:', acc)
    print('Mcc Score:', mcc)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1_score:', f1)
    print('classification_report:: ', classification_report(y_test, y_pred))
    
    max_vote_df.to_csv(f'{args.output_path}Ensemble-{args.ensemble_type}---test_acc---{acc}.csv', index = False)

    conf_mat = confusion_matrix(y_test,y_pred)
    print(conf_mat)