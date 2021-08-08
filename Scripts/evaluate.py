from operator import index
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, f1_score, accuracy_score, precision_score, recall_score

import dataset
from engine import test_eval_fn
from common import get_parser

parser = get_parser()
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

def test_evaluate(test_df, test_data_loader, model, device):
    y_pred, y_test = test_eval_fn(test_data_loader, model, device)
    acc = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print('Accuracy::', acc)
    print('Mcc Score::', mcc)
    print('Precision::', precision)
    print('Recall::', recall)
    print('F_score::', f1)
    print('classification_report:: ', classification_report(y_test, y_pred))
    test_df['y_pred'] = y_pred
    pred_test = test_df[['text', 'label', 'target', 'y_pred']]
    pred_test.to_csv(f'{args.output_path}test_acc---{acc}.csv', index = False)

    conf_mat = confusion_matrix(y_test,y_pred)
    print(conf_mat)
