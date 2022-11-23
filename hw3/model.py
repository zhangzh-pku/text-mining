import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCnn(nn.Module):
    def __init__(self, embed_num, embed_dim, class_num, kernel_num, kernel_sizes, dropout = 0.5):
        super(TextCnn, self).__init__()
        self.embed = nn.Embedding(num_embeddings=embed_num, embedding_dim=embed_dim)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, kernel_num, (f, embed_dim), padding=(2,0)) for f in kernel_sizes])
        self.fc = nn.Linear(kernel_num * len(kernel_sizes), class_num)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.embed(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logit = self.fc(x)
        return logit