
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# 훈련데이터 : H(x) = w1x1 + w2x2 + w3x3 + b
x1_train = torch.FloatTensor([[73],[93],[89],[96],[73]])
x2_train = torch.FloatTensor([[80],[88],[91],[98],[66]])
x3_train = torch.FloatTensor([[75],[93],[90],[100],[70]])
y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])

# 가중치 w와 편향 b를 선언. 가중치 w도 3개 선언해야함
# 가중치 편향 초기화
w1 = torch.zeros(1, requires_grad = True)
w2 = torch.zeros(1, requires_grad = True)
w3 = torch.zeros(1, requires_grad = True)
b = torch.zeros(1, requires_grad = True)

#optimizer 설정
optimizer = optim.SGD([w1, w2, w3, b], lr = 1e-5)
nb_epochs = 1000000
for epoch in range(nb_epochs +1):
    # H(x) 계산
    hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b

    # cost 계산 (평균 제곱 오차)
    cost = torch.mean((hypothesis - y_train)**2)

    # cost로 H(x) 계산
    # gradient를 0으로 초기화 (추후 backward 해줄 때 중첩해서 더해지는걸 방지하기 위해서)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch %100 == 0:
        print('Epoch {:4d}/{} w1: {:.3f} w2: {:.3f} w3: {:.3f} b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, w1.item(), w2.item(), w3.item(), b.item(), cost.item()))

