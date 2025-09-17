package zinc

import (
	_ "embed"
	"github.com/Jimmy2099/torch/pkg/fmt"
	"github.com/Jimmy2099/torch/testing"
)

var pythonScript = `
import torch
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader

# subset: True, False
# split: 'train', 'val', 'test'

train_dataset = ZINC(root='./dataset/ZINC', subset=%s, split='train')
val_dataset = ZINC(root='./dataset/ZINC', subset=%s, split='val')
test_dataset = ZINC(root='./dataset/ZINC', subset=%s, split='test')
`

func loadDataset(subset bool) {
	subSetString := "False"
	if subset == true {
		subSetString = "True"
	}
	testing.RunPyScript(fmt.Sprintf(pythonScript, subSetString, subSetString, subSetString))
}
