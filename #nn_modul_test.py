#Gradient Descent - nn_modul_test

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1) # 랜덤시드값 설정

# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# 선형 회귀 모델 구현
# 모델을 선언 및 초기화. 단순 선형 회귀이므로 input_dim = 1, output_dim = 1
model = nn.Linear(1,1)

#옵티마이저 정의
optimizer = torch.optim.SGD(model.parameters(),lr = 0.01) #lr :learning rate

nb_epochs = 9000
for epoch in range(nb_epochs):
    # H(x)계산
    prediction = model(x_train)

    # cost 계산
    cost = F.mse_loss(prediction, y_train)#파이토치에서 제공하는 평균 제곱 오차 함수

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
    # 100번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))

new_var = torch.FloatTensor([4.0])
pred_y = model(new_var)
print(pred_y)

print(list(model.parameters()))
