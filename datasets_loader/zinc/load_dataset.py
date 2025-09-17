import torch
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader

# subset: True, False
# split: 'train', 'val', 'test'

train_dataset = ZINC(root='./dataset/ZINC', subset=%s, split='train')
val_dataset = ZINC(root='./dataset/ZINC', subset=%s, split='val')
test_dataset = ZINC(root='./dataset/ZINC', subset=%s, split='test')