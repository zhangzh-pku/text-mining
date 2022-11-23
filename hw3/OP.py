import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F


def train(train_iter, dev_iter, model, learning_config):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_config['lr'])
    steps = 0
    best_acc = 0
    for _ in range(learning_config['epochs']):
        for batch in train_iter:
            feature, target = batch.text, batch.label
            feature.t_(), target.sub_(1)
            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % 100 == 0:
                corrects = (torch.max(logit, 1)[1].view(
                    target.size()).data == target.data).sum()
                accuracy = 100.0 * float(corrects) / batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.3f}%({}/{})'.format(
                        steps, loss.data, accuracy, corrects,
                        batch.batch_size))
                if steps % 200 == 0:
                    dev_acc = test(dev_iter, model)
                    if dev_acc > best_acc:
                        best_acc = dev_acc
                        save(model, 'snapshot_output', 'best', steps)
                if steps % 1000 == 0:
                    save(model, 'snapshot_output', 'snapshot', steps)


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir,
                             '{}_steps_{}.pt'.format(save_prefix, steps))
    torch.save(model.state_dict(), save_path)


def test(data_iter, model):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.t_(), target.sub_(1)
        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)
        avg_loss += loss.data
        corrects += (torch.max(logit, 1)[1].view(
            target.size()).data == target.data).sum()
    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * float(corrects) / size
    print('Evaluation - loss: {:.6f}  acc: {:.3f}% ({}/{}) \n'.format(
        avg_loss, accuracy, corrects, size))
    return accuracy