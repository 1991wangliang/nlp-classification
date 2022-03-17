# pip3 install hydra-core omegaconf regex bitarray sacrebleu
import torch

roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
roberta.eval()

tokens = roberta.encode('Hello world!')
assert tokens.tolist() == [0, 31414, 232, 328, 2]
assert roberta.decode(tokens) == 'Hello world!'
print(tokens)

tokens = roberta.encode('你好!')
print(tokens)
tokens = roberta.encode('你好吗!')
print(tokens)
