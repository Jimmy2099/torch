#!pip install -q -U immutabledict sentencepiece 
#!git clone https://github.com/google/gemma_pytorch.git

import contextlib
import torch
import os
import kagglehub

import sys 
sys.path.append("./gemma_pytorch/") 

from gemma.config import get_model_config
from gemma.model import GemmaForCausalLM

# Choose variant and machine type
VARIANT = '1b'
MACHINE_TYPE = 'cpu'
OUTPUT_LEN = 200
METHOD = 'it'

weights_dir = kagglehub.model_download(f"google/gemma-3/pytorch/gemma-3-{VARIANT}-{METHOD}/1")
tokenizer_path = os.path.join(weights_dir, 'tokenizer.model')
ckpt_path = os.path.join(weights_dir, f'model.ckpt')

# Set up model config.
model_config = get_model_config(VARIANT)
model_config.dtype = "float32" if MACHINE_TYPE == "cpu" else "float16"
model_config.tokenizer = tokenizer_path

@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)
    
from torchsummary import summary
import numpy as np

# Instantiate the model and load the weights.
device = torch.device(MACHINE_TYPE)
with _set_default_tensor_type(model_config.get_dtype()):
    model = GemmaForCausalLM(model_config)
    model.load_weights(ckpt_path)
    model = model.to(device).eval()
    # summary(model, input_size=(3, 224, 224))
    print(model)
    for name, param in model.named_parameters():
        if name=="embedder.weight":
            continue
        print(f"Layer: {name}, Shape: {param.shape}")
        np.savetxt("./data/"+name+".csv", param.detach().numpy(), delimiter=",", fmt="%.16f")
    exit(0)
    for name, param in model.named_parameters():
        if name=="embedder.weight":
            continue
        print(f"Layer: {name}, Shape: {param.shape}")
        np_data=model.model.layers[0].self_attn.qkv_proj.weight.numpy()
        print(np_data.shape)
        # np_data = np_data[:100, :100]
        print(len(np_data))
        np.savetxt("fc1_weight.csv", np_data, delimiter=",")
        exit(0)
    exit(0)
# Generate
USER_CHAT_TEMPLATE = "<start_of_turn>user\n{prompt}<end_of_turn>\n"
MODEL_CHAT_TEMPLATE = "<start_of_turn>model\n{prompt}<end_of_turn>\n"

print(model.generate(
    USER_CHAT_TEMPLATE.format(prompt="What is a good place for travel in the US?") +
    MODEL_CHAT_TEMPLATE.format(prompt="California.") +
    USER_CHAT_TEMPLATE.format(prompt="What can I do in California?") +
    "<start_of_turn>model\n",
    device,
    output_len=OUTPUT_LEN
))
