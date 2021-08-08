import torch
import torch.nn as nn
from tqdm import tqdm
import time
import numpy as np
from sklearn.metrics import f1_score

import utils
from common import get_parser

parser = get_parser()
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

def loss_fn(output, target):
    return nn.CrossEntropyLoss()(output, target)


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    losses = utils.AverageMeter()
    progrss_bar = tqdm(data_loader, total = len(data_loader))
    start = time.time()
    train_losses = []
    final_target = []
    final_output = []

    for ii, data in enumerate(progrss_bar):
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]
        token_type_ids = data["token_type_ids"]
        target = data["target"]

        input_ids = input_ids.to(device, dtype = torch.long)
        attention_mask = attention_mask.to(device, dtype = torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        target = target.to(device, dtype=torch.long)

        model.zero_grad()

        output = model(input_ids=input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        loss = loss_fn(output, target)
        train_losses.append(loss.item())
        output = torch.log_softmax(output, dim = 1)
        output = torch.argmax(output, dim = 1)
        end = time.time()
        
        # if(ii%100 == 0 and ii!=0) or (ii == len(data_loader)-1):
        #     print((f'ii={ii}, Train F1={f1},Train loss={loss.item()}, time={end-start}'))

        loss.backward() # Calculate gradients based on loss
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step() # Adjust weights based on calculated gradients
        scheduler.step() # Update scheduler
        losses.update(loss.item(), input_ids.size(0))
        progrss_bar.set_postfix(loss = losses.avg)
        final_target.extend(target.cpu().detach().numpy().tolist())
        final_output.extend(output.cpu().detach().numpy().tolist())
    f1 = f1_score(final_target, final_output, average='weighted')
    f1 = np.round(f1.item(), 4)
    return f1, np.mean(train_losses)

def eval_fn(data_loader, model, device):
    model.eval()
    start = time.time()
    val_losses = []
    final_target = []
    final_output = []

    with torch.no_grad():
        for ii, data in enumerate(data_loader):
            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            token_type_ids = data["token_type_ids"]
            target = data["target"]
            
            input_ids = input_ids.to(device, dtype=torch.long)
            attention_mask = attention_mask.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            target = target.to(device, dtype=torch.long)

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            loss = loss_fn(output, target)
            output = torch.log_softmax(output, dim = 1)
            output = torch.argmax(output, dim = 1)
            val_losses.append(loss.item())
            final_target.extend(target.cpu().detach().numpy().tolist())
            final_output.extend(output.cpu().detach().numpy().tolist())
    f1 = f1_score(final_target, final_output, average='weighted')
    f1 = np.round(f1.item(), 4)
    return f1, np.mean(val_losses)

def test_eval_fn(data_loader, model, device):
    model.eval()
    start = time.time()
    val_losses = []
    final_target = []
    final_output = []

    with torch.no_grad():
        for ii, data in enumerate(data_loader):
            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            token_type_ids = data["token_type_ids"]
            target = data["target"]
            
            input_ids = input_ids.to(device, dtype=torch.long)
            attention_mask = attention_mask.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            target = target.to(device, dtype=torch.long)

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            loss = loss_fn(output, target)
            output = torch.log_softmax(output, dim = 1)
            output = torch.argmax(output, dim = 1)
            val_losses.append(loss.item())
            final_target.extend(target.cpu().detach().numpy().tolist())
            final_output.extend(output.cpu().detach().numpy().tolist())
    print(f'Output length --- {len(final_output)}, Prediction length --- {len(final_target)}')
    return final_output, final_target