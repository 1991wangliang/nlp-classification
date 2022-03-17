# pip3 install torch torchvision torchaudio torchtext torchdata

# https://pytorch.org/text/stable/tutorials/sst2_classification_non_distributed.html
# https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html

import torch
import torch.nn as nn
import time
import os
from torchtext.models import RobertaClassificationHead, XLMR_BASE_ENCODER
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

padding_idx = 1

text_transform = XLMR_BASE_ENCODER.transform()


from torchtext.datasets import SST2


batch_size = 8

# 0 negative
# 1 positive
num_classes = 16
labels = ["negative", "positive"]

train_datapipe = SST2(split='train')
dev_datapipe = SST2(split='dev')
test_datapipe = SST2(split='test')


# Transform the raw dataset using non-batched API (i.e apply transformation line by line)
train_datapipe = train_datapipe.map(lambda x: (text_transform(x[0]), x[1]))
train_datapipe = train_datapipe.batch(batch_size)
train_datapipe = train_datapipe.rows2columnar(["token_ids", "target"])
train_dataloader = DataLoader(train_datapipe, batch_size=None)

dev_datapipe = dev_datapipe.map(lambda x: (text_transform(x[0]), x[1]))
dev_datapipe = dev_datapipe.batch(batch_size)
dev_datapipe = dev_datapipe.rows2columnar(["token_ids", "target"])
dev_dataloader = DataLoader(dev_datapipe, batch_size=None)


input_dim = 768

classifier_head = RobertaClassificationHead(num_classes=num_classes, input_dim=input_dim)

model = XLMR_BASE_ENCODER.get_model(head=classifier_head)
# use multiple GPUs to train
model = nn.DataParallel(model)
model.to(DEVICE)

import torchtext.functional as F
from torch.optim import AdamW


learning_rate = 1e-5
optim = AdamW(model.parameters(), lr=learning_rate)
criteria = nn.CrossEntropyLoss()


def train_step(input, target):
    output = model(input)
    loss = criteria(output, target)
    optim.zero_grad()
    loss.backward()
    optim.step()


def eval_step(input, target):
    output = model(input)
    loss = criteria(output, target).item()
    return float(loss), (output.argmax(1) == target).type(torch.float).sum().item()


def evaluate():
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    counter = 0
    with torch.no_grad():
        for batch in dev_dataloader:
            input = F.to_tensor(batch['token_ids'], padding_value=padding_idx).to(DEVICE)
            target = torch.tensor(batch['target']).to(DEVICE)
            loss, predictions = eval_step(input, target)
            total_loss += loss
            correct_predictions += predictions
            total_predictions += len(target)
            counter += 1

    return total_loss / counter, correct_predictions / total_predictions


def train(num_epochs=1):
    best = 0.0
    for e in range(num_epochs):
        t1 = time.time()
        for batch in train_dataloader:
            input = F.to_tensor(batch['token_ids'], padding_value=padding_idx).to(DEVICE)
            target = torch.tensor(batch['target']).to(DEVICE)
            train_step(input, target)

        loss, accuracy = evaluate()
        if accuracy > best:
            best = accuracy
            # save models
            torch.save(model.state_dict(), 'best.pth')
            print("save best model loss = [{}], accuracy = [{}]".format(loss, accuracy))

        t2 = time.time()
        print("Epoch = [{}],time = {} , loss = [{}], accuracy = [{}]".format(e, (t2 - t1), loss, accuracy))
        torch.save(model.state_dict(), 'Epoch[{}].pth'.format(e))


if os.path.exists('best.pth'):
    print('load best model.')
    model.load_state_dict(torch.load('best.pth', map_location=DEVICE))
    model.eval()
else:
    print('train num_epochs 10')
    train(10)

print(model)

# test model
input_batch = ["the emotions are raw and will strike a nerve with anyone who 's ever had family trauma . "]
model_input = F.to_tensor(text_transform(input_batch), padding_value=padding_idx)

# output add softmax function
softmax = nn.Softmax(dim=1)
output = softmax(model(model_input))

print(output)
# print result classification index
index = output.argmax(1)
print(index)

# assert result
assert index == 1, 'train result was fail .'

# test data
for x in test_datapipe:
    print("text:{}".format(x[0]))
    model_input = F.to_tensor(text_transform([x[0]]), padding_value=padding_idx)
    output = model(model_input)
    output = softmax(output)
    index = output.argmax(1)
    print("text predict classification:{},score:{}".format(labels[int(index[0])], float(output[0][index])))
