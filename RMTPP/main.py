import torch
import torch.cuda
import torch.optim as optim
import numpy as np
from model import load_model
from dataset import ATMDataset
import matplotlib.pyplot as plt

def log_evaluation(target_time, predicted_time, target_events, predicted_events):
    tmp1 = map(abs, np.array(target_time) - np.array(predicted_time))
    tmp2 = map(abs, np.array(target_events) - np.array(predicted_events))
    print("average_time_diff", sum(tmp1) / len(target_time))
    print("average_event_diff", sum(tmp2) / len(target_time))

def main():
    device = torch.device('cpu')
    batchsize = 4
    dataset = ATMDataset(dir='data', device=device)
    train_iter = dataset.train_iter(batch_size=batchsize)
    test_iter = dataset.test_iter(batch_size=batchsize)

    model = load_model()
    model.to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    train(model, optimizer, scheduler, train_iter, test_iter, 200)

    target_time, predicted_time, target_events, predicted_events = evaluate(model, test_iter)

    print(target_time)
    print(predicted_time)
    print(target_events)
    print(predicted_events)

    log_evaluation(target_time, predicted_time, target_events, predicted_events)


def evaluate(model, data_iter):
    predicted_events = []
    target_events = []
    predicted_time = []
    target_time = []
    model.eval()
    for index, (input, target) in enumerate(data_iter):
        time, event = model.generate(input)
        predicted_time.extend(time.cpu().numpy().tolist())
        target_time.extend(target[0].tolist())
        predicted_events.extend(event.cpu().numpy().tolist())
        target_events.extend(target[1].tolist())
    log_evaluation(target_time, predicted_time, target_events, predicted_events)
    return target_time, predicted_time, target_events, predicted_events


def train(model, optimizer, scheduler, train_iter,test_iter, epochs):

    tLoss = []
    for epoch in range(epochs):
        model.train()
        print("nepoch:", epoch)
        scheduler.step()
        train_iter.shuffle()
        losses_list = [[], [], []]
        for index, (input, target) in enumerate(train_iter):
            model.zero_grad()
            time_loss, event_loss, merged_loss = model.loss(input, target)
            merged_loss.backward()
            optimizer.step()
            losses_list[0].append(time_loss.item())
            losses_list[1].append(event_loss.item())
            losses_list[2].append(merged_loss.item())
            tLoss.append(time_loss.item())


            if index * train_iter.batch_size * 5 // len(train_iter) != (index - 1) * train_iter.batch_size * 5 // len(train_iter):
                print("now loss:", np.mean(losses_list[0]), np.mean(losses_list[1]), np.mean(losses_list[2]))
                losses_list = [[], [], []]

        target_time, predicted_time, target_events, predicted_events = evaluate(model, test_iter)
        print("now evaluate:")
        print(target_time, predicted_time, target_events, predicted_events)


    print(tLoss)
    plt.plot(range(len(tLoss)), tLoss)
    plt.show()

if __name__ == "__main__":
    main()