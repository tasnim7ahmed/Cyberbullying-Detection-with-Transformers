import pandas as pd;
import numpy as np;
import torch
from transformers import BertModel, AdamW, get_scheduler
from collections import defaultdict
from sklearn.metrics import f1_score
import warnings

import engine
from model import BertFGBC
from dataset import Dataset
from utils import train_validate_test_split
from common import get_parser
from evaluate import test_evaluate

parser = get_parser()
args = parser.parse_args()
warnings.filterwarnings("ignore")
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

def run():
    df = pd.read_csv(args.dataset_file).dropna()
    print(set(df.label.values))
    
    train_df, valid_df, test_df = train_validate_test_split(df)

    print("train len - {}, valid len - {}, test len - {}".format(len(train_df), len(valid_df),len(test_df)))

    train_dataset = Dataset(text=train_df.text.values, target=train_df.target.values)
    train_data_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = args.train_batch_size,
        shuffle = True
    )

    valid_dataset = Dataset(text=valid_df.text.values, target=valid_df.target.values)
    valid_data_loader = torch.utils.data.DataLoader(
        dataset = valid_dataset,
        batch_size = args.valid_batch_size,
        shuffle = True
    )

    test_dataset = Dataset(text=test_df.text.values, target=test_df.target.values)
    test_data_loader = torch.utils.data.DataLoader(
        dataset = test_dataset,
        batch_size = args.test_batch_size,
        shuffle = False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertFGBC()
    model = model.to(device)

    num_train_steps = int(len(train_df) / args.train_batch_size * args.epochs)
    
    optimizer = AdamW(
        params = model.parameters(),
        lr = args.learning_rate,
        weight_decay = args.weight_decay,
        eps = args.adamw_epsilon
    )

    scheduler = get_scheduler(
        "linear",
        optimizer = optimizer,
        num_warmup_steps = args.warmup_steps,
        num_training_steps = num_train_steps
    )

    print("---Starting Training---")

    history = defaultdict(list)
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}/{args.epochs}')
        print('-'*10)

        train_acc, train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        print(f'Epoch {epoch + 1} --- Training loss: {train_loss} Training accuracy: {train_acc}')
        val_acc, val_loss = engine.eval_fn(valid_data_loader, model, device)
        print(f'Epoch {epoch + 1} --- Validation loss: {val_loss} Validation accuracy: {val_acc}')
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc>best_acc:
            torch.save(model.state_dict(), f"{args.model_path}{args.pretrained_model_name}---val_acc---{val_acc}.bin")

    print("##################################### Testing ############################################")
    test_evaluate(test_df, test_data_loader, model, device)    
    del model, train_data_loader, valid_data_loader, train_dataset, valid_dataset
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("##################################### Task End ############################################")

if __name__=="__main__":
    run()