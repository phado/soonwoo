from numpy import require
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x_train = torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  80], 
                               [96,  98,  100],   
                               [73,  66,  70]]) 
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

print(x_train.shape)
print(y_train.shape)

# 가중치와 편향 선언
w = torch.zeros((3,1), requires_grad = True)
b = torch.zeros(1, requires_grad = True)
optimizer = optim.SGD([w, b], lr=1e-5)

hypothesis = x_train.matmul(w)+b

nb_epochs = 20
for epoch in range(nb_epochs +1):
    # 편향 b는 브로드 캐스팅 되어 각 샘플에 더해짐
    hypothesis = x_train.matmul(w) + b

    cost = torch.mean((hypothesis - y_train)**2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
        epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()
    ))

