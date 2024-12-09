# from torchsummary import summary
import torch
from torchinfo import summary
import torchvision.models as models
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

module1=models.alexnet()
module1=module1.to(device=device)

module2=models.vgg16()
module2=module2.to(device=device)

module3=models.resnet18()
module3=module3.to(device=device)

module4=models.efficientnet_b5()
module4=module4.to(device=device)

module5=models.resnext50_32x4d()
module5=module5.to(device=device)

module6=models.densenet121()
module6=module6.to(device=device)

module7=models.mobilenet_v3_large()
module7=module7.to(device=device)


#1，512，512
summary(module1, (1, 3, 512, 512))
summary(module2, (1, 3, 224, 224))
summary(module3, (1, 3, 224, 224))
summary(module4, (1, 3, 224, 224))
summary(module5, (1, 3, 224, 224))
summary(module6, (1, 3, 224, 224))
summary(module7, (1, 3, 224, 224))




def param(model):
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0
    # 遍历model.parameters()返回的全局参数列表
    for param in model.parameters():
        mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
        Total_params += mulValue  # 总参数量
        if param.requires_grad:
            Trainable_params += mulValue  # 可训练参数量
        else:
            NonTrainable_params += mulValue  # 非可训练参数量

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')

param(module1)
param(module2)
param(module3)
param(module4)
param(module5)
param(module6)
param(module7)
