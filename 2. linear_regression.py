import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable

# 입력 데이터 수
num_data = 10

# x, noise가 섞인 y 데이터 생성
x = init.uniform(torch.Tensor(num_data,1),-10,10)
noise = init.normal(torch.FloatTensor(num_data,1), std=0.2)
y_noise = 2*(x+noise)+3

# 모델 생성
model = nn.Linear(1,1)

# loss function 정의
loss_func = nn.L1Loss()

# optimizer 정의
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 실제 y값(label)
label = Variable(y_noise)

# training
loss_arr = []
for i in range(1000):
    output = model(Variable(x))
    optimizer.zero_grad()
    loss = loss_func(output,label)
    loss.backward()
    optimizer.step()
    loss_arr.append(loss.data.numpy()[0])

loss_arr
list(model.parameters())
