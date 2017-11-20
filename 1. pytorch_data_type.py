import torch

# 2x3의 FloatTensor 생성
torch.rand(2, 3)
torch.randn(2, 3)
torch.zeros(2, 3)
torch.ones(2, 3)
torch.FloatTensor(2, 3)

# [ ]인 FlaotTensor 생성
torch.FloatTensor([2, 3])

# 5개 LongTensor 생성
torch.randperm(5)

# torch.arange(start,end,step=1) -> [start,end) with step
torch.arange(0, 3, step=0.5)

# torch.numel(torch.zeros(4,4)) 요소의 갯수 리턴

import numpy as np
a = np.ndarray(shape=(2, 3), dtype=int, buffer=np.array([1.0,2.0,3.0,4.0,5.0,6.0]))

# LongTensor to (DoubleTensor, FlaotTensor)
## Pytorch data type link : http://pytorch.org/docs/master/tensors.html
torch.from_numpy(a).double()
torch.from_numpy(a).float()

# GPU 사용
torch.FloatTensor([[1,2,3],[4,5,6]]).cuda()
torch.cuda.FloatTensor([[6,5,4],[3,2,1]])

# tensor size
torch.FloatTensor(10,12,3,3).size()

# indexing
x = torch.rand(4, 3)
torch.index