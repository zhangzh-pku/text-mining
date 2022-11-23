import json
import torch
import torch.nn.functional as F
from torchtext import data
import jieba
from model import TextCnn
import OP

if __name__ == "__main__":
    f = open('config.json', 'r')
    text = f.read()
    config = json.loads(text)
    model_config = config['model']
    learning_config = config['learning']
    print(config)
    def tokenize(text):
        return [word for word in jieba.cut(text) if word.strip()]

    text_field = data.Field(lower=True, tokenize=tokenize)
    label_field = data.Field(sequential=False)
    fields = [('text', text_field), ('label', label_field)]
    train_dataset, test_dataset = data.TabularDataset.splits(path='./data/',
                                                             format='tsv',
                                                             skip_header=False,
                                                             train='train.tsv',
                                                             test='test.tsv',
                                                             fields=fields)
    text_field.build_vocab(train_dataset,
                           test_dataset,
                           min_freq=5,
                           max_size=50000)
    label_field.build_vocab(train_dataset, test_dataset)
    train_iter, test_iter = data.Iterator.splits(
        (train_dataset, test_dataset),
        batch_sizes=(learning_config['batch_size'],
                     learning_config['batch_size']),
        sort_key=lambda x: len(x.text))
    embed_num = len(text_field.vocab)
    class_num = len(label_field.vocab) - 1
    kernel_size = [int(k) for k in model_config['kernel_size'].split(',')]
    cnn = TextCnn(embed_num=embed_num,
                  embed_dim=model_config['embed_dim'],
                  class_num=class_num,
                  kernel_num=model_config['kernel_num'],
                  kernel_sizes=kernel_size,
                  dropout=model_config['dropout'])
    
    OP.train(train_iter, test_iter, cnn, learning_config)
    OP.test(test_iter, cnn)
