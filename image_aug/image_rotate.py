import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
print('x_data:\n',x_data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print('np_array:\n',np_array)
print('x_np:\n',x_np)

x_ones = torch.ones_like(x_data) #x_data의 속성을 유지합니다
print(f'ones Tendor:\n{x_ones} \n')
x_rand = torch.rand_like(x_data, dtype = torch.float) # x_data의 속성을 덮어씁니다
print(f'Random Tensor:\n{x_rand}\n')

#shape은 텐서의 차원을 나타내는 튜플로 아래 함수들에서는 출력 텐서의 차원을 결정함
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f'Random Tensor : \n{rand_tensor}\n')
print(f'One Tensor : \n{ones_tensor}\n')
print(f'Zero Tensor : \n{zeros_tensor}\n')

tensor = torch.rand(3,4)
print(f'Shape of tensor : {tensor.shape}')
print(f'Datatype of tensor :{tensor.dtype}')
print(f'Device of tensor : {tensor.device}')

#GPU가 존재하면 텐서를 이동합니다
if torch.cuda.is_available():
    tensor = tensor.to('cuda')
    print(f'Device tensor is stored on: {tensor.device}')

tensor = torch.ones(4,4)
tensor[:,1] = 0
print(tensor)


t1 = torch.cat([tensor, tensor, tensor],dim=1)
print(t1)

#텐서의 곱
print(f'tensor.mul(tensor)\n{tensor.mul(tensor)}\n')


print(f'tensor.matmul(tensor.T)\n{tensor.matmul(tensor.T)}\n')
print(f'tensor @ tendor.T \n {tensor @ tensor.T}')

t = np.array([0., 1., 2., 3., 4., 5., 6.])
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])



t=torch.FloatTensor([[1,2,3],
                     [4,5,6],
                     [7,8,9],
                     [10,11,12]])
t1 = torch.arange(12).view(3,4)

print(t[1:,2])
print(t[1:2])
print(t[:,1])
print(t[:,:-1])

#pytorch의 broadcasting
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3], [4]])
print(m1 + m2)
print(m2+m1)


m3 = torch.FloatTensor([[1,2],[3,4]])
m4 = torch.FloatTensor([[2,3],[4,5]])
print('Shape of Matrix1 : ', m3.shape)#2x2
print('Shape of Matrix2 : ', m4.shape)#2x1
print(m3.matmul(m4))#2x1


t = torch.FloatTensor([[1,2],[3,4]])
print(t.max(dim=0))


w = torch.tensor(1.0,requires_grad=True)
y= w**2
z=2*y+5
z.backward()
print(w.grad)


x = torch.ones(3, requires_grad= True)
y= x**2
z= y**2+x
out = z.sum()
grad = torch.Tensor([0.1, 1, 100])
z.backward(grad)
print(x.data)
print(x.grad)
print(x.grad)


input = torch.randn(2,2,2)
input1 = torch.rand(2,2)

print(input)
print(input1)
m = nn.Conv2d(16, 33, 3, stride = 2)
m = nn.Conv2d(16, 33, (3, 5), stride = (2,1), padding = (4,2))
m = nn.Conv2d(16, 33, (3, 5), stride = (2,1), padding = (4,2), dilation = (3, 1))

input = torch.randn(20, 16, 50, 100)
output = m(input)
output.shape
print('-------------------------------------------------')
torch.manual_seed(1)
#데이터
x_train = torch.FloatTensor([[1],[2],[3]])
y_trina = torch.FloatTensor([[2],[4],[6]])

#모델을 선언 및 초기화. 단순 선형회귀 이므로 input_dim = 1, output_dim=1
model = nn.Linear(1,1)
print(list(model.parameters()))

#optimizer 설정. 경사 하강법 SGD를 사용하고 learning rate를 의미하는 lr은 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#전체 훈련 데이터에 대해 경사 하강법을 2,000회 반복
nb_epochs = 2000
for epoch in range(nb_epochs+1):
    #H(x) 계산
    prediction = model(x_train)

    #cost 계산
    cost = F.mse_loss(prediction, y_trina) #파이토치에서 제공하는 평균 제곱 오차 함수

    #cost로 H(x) 개선하는 부분
    #gradient를 0으로 초기화 
    optimizer.zero_grad()
    #비용함수를 미분하여 gradient계산
    cost.backward()#backward 연산
    #W와 b를 업데이트
    optimizer.step()

    if epoch %100 ==0:
        #100번마다 로그 출력
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))