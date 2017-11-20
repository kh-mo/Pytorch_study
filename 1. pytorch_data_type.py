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
torch.index_select(x, 1, torch.LongTensor([0, 1])) # torch.index_select(data, 행/열, index)
x[:,0]
x[0,:]
x[0:2,0:2]
x = torch.randn(2, 3)
mask = torch.ByteTensor([[0,0,1],[0,1,0]])
out = torch.masked_select(x,mask)

# joining
x = torch.FloatTensor([[1,2,3],[4,5,6]])
y = torch.FloatTensor([[-1,-2,-3],[-4,-5,-6]])
z1 = torch.cat([x,y],dim=0)
z2 = torch.cat([x,y],dim=1)
x = torch.FloatTensor([[1,2,3],[4,5,6]])
x_stack = torch.stack([x,x,x,x], dim=0)

# slicing
torch.chunk(z1, 1, dim=0) # torch.chunk(data, 자를 chunk 수/만들 chunk 수, dimension)
torch.split(z1, 2, dim=0) # torch.split(data, 자를 split 수/만들 split 수, dimension)

# squeezing - 1크기 dimension 제거
x1 = torch.FloatTensor(10,1,3,1,4)
x2 = torch.squeeze(x1)
x1.size(),x2.size()

# unsqueezing - 1크기 dimension 추가
x1 = torch.FloatTensor(10,3,4)
x2 = torch.unsqueeze(x1,dim=0)
x1.size(),x2.size()


# 분포로부터 데이터 생성
import torch.nn.init as init
init.uniform(torch.FloatTensor(3,4),a=0,b=9) # a~b까지 uniform distribution에서 데이터 생성
init.normal(torch.FloatTensor(3,4),mean=0,std=1) # 평균이 mean이고 표준편차가 std인 normal distribution에서 데이터 생성
init.constant(torch.FloatTensor(3,4), val=5) # val 값으로 input 데이터 크기의 상수 생성

# 수학 연산
a = torch.FloatTensor([[1,2,3],[4,5,6]])
b = torch.FloatTensor([[1,2,3],[4,5,6]])

torch.add(a,b)
torch.add(a,10) # broadcasting
a+b

torch.mul(a,b)
torch.mul(a,10) # broadcasting
a*b

torch.div(a,b)
a/b

torch.pow(a,4) # 지수 승
a**4

torch.exp(a) # e의 지수 승

torch.log(a) # log

# matrix operations
a = torch.zeros(3,4)
b = torch.zeros(4,5)
torch.mm(a,b)

a = torch.FloatTensor(10,3,4)
b = torch.FloatTensor(10,4,5)
torch.bmm(a,b).size() # batch단위 matrix operation

# 내적 계산
a = torch.FloatTensor([3,4])
b = torch.FloatTensor([3,4])
torch.dot(a,b)

# transposed matrix
torch.FloatTensor(3,4)
torch.FloatTensor(3,4).t()
torch.FloatTensor(10,3,4).size()
torch.transpose(torch.FloatTensor(10,3,4),1,2).size()
torch.FloatTensor(10,3,4).transpose(1,2).size()

# eigen_value, eigen_vector
a = torch.FloatTensor(4,4)
torch.eig(a, eigenvectors=True)
