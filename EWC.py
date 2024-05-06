from LSTM import LSTM
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LSTM(2, len(vocab), 256, 256)

lr=3e-5

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def test(testLoader, model, optimizer):
    print('Testing')
    batch_acc = []
    h0, c0 =  model.init_hidden(batch_size=50)
    for batch_idx, batch in enumerate(testLoader):

        input = batch[0].to(device)
        target = batch[1].to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            out, hidden = model(input, (h0, c0))
            _, preds = torch.max(out, 1)
            preds = preds.to("cpu").tolist()
            batch_acc.append(accuracy_score(preds, target.tolist()))

    accuracy = sum(batch_acc)/len(batch_acc)
    return accuracy

def update_fisher(task_id, dataloader):

    model.train()
    optimizer.zero_grad()
  
    for i, data in enumerate(dataloader, 0):      
        inputs, labels = data        
        output = model(inputs)
        loss = F.cross_entropy(output, labels)
        loss.backward()

    fisher_dict[task_id] = {}
    optpar_dict[task_id] = {}

    for name, param in model.named_parameters():
    
        optpar_dict[task_id][name] = param.data.clone()
        fisher_dict[task_id][name] = param.grad.data.clone().pow(2)

def train_ewc(model, task_id, dataloader, optimizer, epochs, language, testLoader):
    print(language)
    accuracy_array = []
    for epoch in range(epochs):
        model.train()
        h0, c0 =  model.init_hidden(batch_size=50)
        for batch_idx, batch in enumerate(dataloader, 0):
            inputs = batch[0].to(device)
            labels = batch[1].to(device)    
            optimizer.zero_grad()
            
            output, hidden = model(inputs, (h0, c0))
            loss = loss_fn(output, labels.long())
            for task in range(task_id):
                for name, param in model.named_parameters():
                    fisher = fisher_dict[task][name]
                    optpar = optpar_dict[task][name]
                    loss += (fisher * (optpar - param).pow(2)).sum() * ewc_lambda
            loss.backward()
            optimizer.step()

        batch_acc = []
        for batch_idx, batch in enumerate(testLoader):

            input = batch[0].to(device)
            target = batch[1].to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                out, hidden = model(input, (h0, c0))
                _, preds = torch.max(out, 1)
                preds = preds.to("cpu").tolist()
                batch_acc.append(accuracy_score(preds, target.tolist()))

        accuracy = sum(batch_acc)/len(batch_acc)
        print(accuracy)
        accuracy_array.append(accuracy)


fisher_dict = {}
optpar_dict = {}
ewc_lambda = 0.01