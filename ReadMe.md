# 文档
https://www.yuque.com/qcwlmqy/rv9v9h/uvusw3gp3sfgtv9x?singleDoc# 《Resnet101 自用》

# 参考
迁移学习的简单知识：
迁移学习有两类比较主要的应用场景

- 将预训练模型作为初始化的参数，替换分类器后，训练和微调整个网络的数据
- 将预训练模型（删除最后一个全连接层）作为固定特征提取器，仅训练一个线性分类器

一般而言，数据集较大的时推荐使用前一种策略，后一种推荐用于小数据集
resnet101
# 代码

# 实验
## 1、实验环境
如下为我创建环境用的命令，创建一个3.10 的python环境安装pytorch、numpy、matplotlib以及pytorch-lightning
```cpp
conda create -n binary python=3.10
conda activate binary
# only cpu pytorch
pip install torch torchvision torchaudio
# gpu pytorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install matplotlib
pip install pytorch-lightning
```
当然你可以在我共享的代码中，找到 environment.yml，直接执行如下代码，一键生成环境
![image.png](https://cdn.nlark.com/yuque/0/2023/png/32590345/1683791213828-52d0414d-63f8-4c03-8120-55d3979736d3.png#averageHue=%23363e43&clientId=ud2825136-9429-4&from=paste&height=133&id=u34a4829d&originHeight=133&originWidth=469&originalType=binary&ratio=1&rotation=0&showTitle=false&size=9048&status=done&style=none&taskId=u049b55ca-cdad-4762-8b38-b78af54ccdb&title=&width=469)
```cpp
conda env create -f environment.yml -n binary
```
## 2、实验数据集
这是一份关于蚂蚁和蜜蜂的非常小的数据集（少于200张），是ImageNet的一小部分
下载地址
数据集结构类似：
![image.png](https://cdn.nlark.com/yuque/0/2023/png/32590345/1683788124953-ddddcbd4-4603-4789-afe1-b11cd2c6061a.png#averageHue=%23131313&clientId=ud2825136-9429-4&from=paste&height=497&id=u539cbc91&originHeight=497&originWidth=716&originalType=binary&ratio=1&rotation=0&showTitle=false&size=39144&status=done&style=none&taskId=ue945cc36-db26-44da-a278-490f0aa9e29&title=&width=716)
训练的蜜蜂数据集：
![image.png](https://cdn.nlark.com/yuque/0/2023/png/32590345/1683791033995-1583f87e-9a65-4a17-8470-ea18e8546d5b.png#averageHue=%23dbd4c5&clientId=ud2825136-9429-4&from=paste&height=515&id=u446af080&originHeight=515&originWidth=954&originalType=binary&ratio=1&rotation=0&showTitle=false&size=408266&status=done&style=none&taskId=ue44dc3d4-7162-41a9-a9c4-9bb38b53643&title=&width=954)
## 3、数据集代码
首先定义两个数据变换的组合
为什么要这么预处理：[https://pytorch.org/vision/main/models/generated/torchvision.models.resnet101.html#torchvision.models.ResNet101_Weights](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet101.html#torchvision.models.ResNet101_Weights)
对于训练集：

1. transforms.RandomResizedCrop(224)：此函数对输入图像进行随机裁剪，裁剪后的图像尺寸是（224，224）。它在从图像中随机选定一个矩形区域后执行。
2. transforms.RandomHorizontalFlip()：此函数随机翻转输入图像。这可以增加数据的多样性，因为它可以创建更多不同的图像。
3. transforms.ToTensor()：此函数将PIL图像数据类型转换为 PyTorch 张量。
4. transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])：此函数对每个通道应用标准归一化，即将每个通道的值减去其均值 (mean) 并除以其标准差 (standard deviation)。

对于测试集：

1. transforms.Resize(256)：将验证集的图像大小重新调整为256x 256。
2. transforms.CenterCrop(224)：在中央裁剪图像的中央，裁剪出大小为224x 224的部分。
3. transforms.ToTensor()：将每个图像JPEG文件转换为PyTorch张量。
4. transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])：应用与训练集预处理操作中完全相同的标准化。

然后使用 datasets.ImageFolder 类来加载数据集
datasets.ImageFolder 类是 PyTorch 中用于处理图像数据集的类之一。它假定你有一个文件夹，其中包含许多其子文件夹中包含特定类别的图片。它使用该文件夹的路径作为根目录来构建数据集。这个类要求文件夹中的每个子文件夹都对应于不同的类，并且这些子文件夹的名称将被用作类的名称。 transforms 参数可用于应用图像预处理和增强操作。在实例化 ImageFolder 类时，你需要指定数据集所在的文件夹路径和transforms
最后使用 DataLoader 类来对图像数据集进行批次处理，并使用 next(iter(dataloaders['train'])) 取出一个批次的数据，用于查看数据集以及其标签。然后将这个批次的数据转换为网格并显示它们的类名称作为标签。

```python
import os

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torchvision import transforms, datasets


# 获取数据集
def get_dataloaders(data_dir):
    # 定义数据路径
    data_dir = 'data/hymenoptera_data'

    # 定义图像的变换和数据增强 transform
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 使用 ImageFolder 类自动构造分类数据集，会将文件夹名作为分类标签
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    # 构造 dataloader
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    return image_datasets, dataloaders


    # 从tensor 转化为图片并显示
def imshow(inp, title=None):
    # 从 chw -> hwc
    inp = inp.numpy().transpose((1, 2, 0))
    # 还原归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    # 显示图像
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


if __name__ == "__main__":
    image_datasets, dataloaders = get_dataloaders(data_dir='data/hymenoptera_data')
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # 获取类的名称集合
    class_names = image_datasets['train'].classes

    # 查看数据集
    inputs, classes = next(iter(dataloaders['train']))
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])

```
运行结果：
![image.png](https://cdn.nlark.com/yuque/0/2023/png/32590345/1683793486982-609d61f6-64bf-4d98-827a-013ee0164636.png#averageHue=%23eee4dc&clientId=ud2825136-9429-4&from=paste&height=466&id=ub226e19e&originHeight=466&originWidth=642&originalType=binary&ratio=1&rotation=0&showTitle=false&size=110549&status=done&style=none&taskId=u455c68cf-1ece-4e29-bd0b-787748949f0&title=&width=642)
## 4、微调整个网络
我们需要更换符合我们需要的分类器，以二分类为例显示
### 4.1 模型
使用 pytorch lighting的框架定义模型，简而言之使用的 pytorch lighting实现 pl.LightningModule 父类的函数，在调用时pytorch lighting 框架会自动调用 training_step、validation_step 等，而且不需要写对象传递cuda、梯度、loss回传等步骤，以下步骤会被自动执行
你可以在 [https://lightning.ai/docs/pytorch/stable/starter/introduction.html](https://lightning.ai/docs/pytorch/stable/starter/introduction.html) 快速了解它
```python
# put model in train mode and enable gradient calculation
model.train()
torch.set_grad_enabled(True)
for batch_idx, batch in enumerate(train_dataloader):
    loss = training_step(batch, batch_idx)
    # clear gradients
    optimizer.zero_grad()
    # backward
    loss.backward()
    # update parameters
    optimizer.step()
```

1. 导入需要的 PyTorch Lightning、PyTorch 和 torchvision 库。
2. 创建一个 PyTorch Lightning 的 LightningModule 类 ResNet。其中，加载了预训练的 ResNet101 模型，获取分类器的输入特征维度，替换了分类器为一个全连接层，并定义使用交叉熵作为 loss 计算方式。
3. 实现 forward、training_step、validation_step 和 configure_optimizers 四个函数。其中，forward 函数将输入数据传入 ResNet 模型并返回输出结果。training_step 和 validation_step 分别实现了训练和验证每个 batch 的具体操作，包括输入、调用 forward 函数、计算 loss 和准确率等，将计算结果通过 log_dict 函数记录在 tensorboard 日志文件中。configure_optimizers 定义了使用 SGD 优化器和每7轮学习率乘以 0.1 的学习率调节器训练模型。
```python
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import models


# define the LightningModule
class ResNet(pl.LightningModule):
    def __init__(self, num_classes=2):
        super().__init__()
        # 加载resnet101的预训练模型
        # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet101.html#torchvision.models.ResNet101_Weights
        self.model = models.resnet101(weights='IMAGENET1K_V1')
        # 获取分类器的输入特征维度
        num_ftrs = self.model.fc.in_features
        # 替换分类器
        self.model.fc = nn.Linear(num_ftrs, num_classes)

        # loss 计算方式，交叉熵
        self.criterion = nn.CrossEntropyLoss()

        # pytroch_lighting 的 api 会计算 forword 的 summary
        self.example_input_array = torch.Tensor(4, 3, 224, 224)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # 获取 input 和 label
        (inputs, labels) = batch
        # 调用 forword 函数，预测
        outputs = self(inputs)
        # 第一结果是最大的值，第二个结果是最大的下标即预测的结果
        _, preds = torch.max(outputs, dim=1)
        # 计算 loss
        loss = self.criterion(outputs, labels)
        acc = torch.sum(preds == labels.data) * 1.0 / inputs.size()[0]
        # log train_loss
        self.log_dict({"train_loss": loss, "train_acc": acc}, prog_bar=True, logger=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # 获取 input 和 label
        (inputs, labels) = batch
        # 调用 forword 函数，预测
        outputs = self(inputs)
        # 第一结果是最大的值，第二个结果是最大的下标即预测的结果
        _, preds = torch.max(outputs, dim=1)
        # 计算 loss 和 acc
        loss = self.criterion(outputs, labels)
        acc = torch.sum(preds == labels.data) * 1.0 / inputs.size()[0]
        # log train_loss
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        # 所有参数优化
        optimizer_ft = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        # 每7轮学习率乘0.1
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        return [optimizer_ft], [exp_lr_scheduler]

```
### 4.2 训练

1. 调用了两个自定义函数：get_dataloaders和ResNet来构建数据加载器和模型。
2. 在训练过程中，它使用ModelCheckpoint回调函数来保存验证loss最低的模型。 
3. 训练过程使用单个GPU进行加速，并使用32位精度进行计算。
4. 最后，训练函数被调用并传入数据集目录作为参数。
```python
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from exp_dataset import get_dataloaders
from exp_model import ResNet


def train(data_dir):
    # 获取数据集
    datasets, dataloaders = get_dataloaders(data_dir=data_dir)

    # resnet
    resnet = ResNet()
    # checkpoint 保存策略, 保存 验证acc最大的模型，且只保存1个
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode='min', filename='{epoch}-{val_loss:.2f}', save_top_k=1)
    # 使用第0张gpu训练，精度为32位
    trainer = pl.Trainer(accelerator="gpu", devices=[0], precision=32, max_epochs=25, callbacks=[checkpoint_callback])
    # 使用model，训练集和验证集开始训练
    trainer.fit(model=resnet,
                train_dataloaders=dataloaders['train'],
                val_dataloaders=dataloaders['val'])


if __name__ == "__main__":
    train(data_dir='data/hymenoptera_data')
```
训练过程
```python
Epoch 0: 100%|██████████| 61/61 [00:24<00:00,  2.44it/s, v_num=0, train_loss_step=1.310, train_acc_step=0.500]
Validation: 0it [00:00, ?it/s]
Validation:   0%|          | 0/38 [00:00<?, ?it/s]
Validation DataLoader 0:   0%|          | 0/38 [00:00<?, ?it/s]
Validation DataLoader 0:   3%|▎         | 1/38 [00:00<00:00, 40.91it/s]
Validation DataLoader 0:   5%|▌         | 2/38 [00:00<00:05,  7.16it/s]
Validation DataLoader 0:   8%|▊         | 3/38 [00:00<00:04,  8.18it/s]
Validation DataLoader 0:  11%|█         | 4/38 [00:00<00:03, 10.18it/s]
Validation DataLoader 0:  13%|█▎        | 5/38 [00:00<00:02, 11.96it/s]
Validation DataLoader 0:  16%|█▌        | 6/38 [00:00<00:02, 13.54it/s]
Validation DataLoader 0:  18%|█▊        | 7/38 [00:00<00:02, 14.92it/s]
Validation DataLoader 0:  21%|██        | 8/38 [00:00<00:01, 16.16it/s]
Validation DataLoader 0:  24%|██▎       | 9/38 [00:00<00:01, 17.23it/s]
Validation DataLoader 0:  26%|██▋       | 10/38 [00:00<00:01, 18.17it/s]
Validation DataLoader 0:  29%|██▉       | 11/38 [00:00<00:01, 19.05it/s]
Validation DataLoader 0:  32%|███▏      | 12/38 [00:00<00:01, 19.76it/s]
Validation DataLoader 0:  34%|███▍      | 13/38 [00:00<00:01, 20.53it/s]
Validation DataLoader 0:  37%|███▋      | 14/38 [00:00<00:01, 21.24it/s]
Validation DataLoader 0:  39%|███▉      | 15/38 [00:00<00:01, 21.90it/s]
Validation DataLoader 0:  42%|████▏     | 16/38 [00:00<00:00, 22.52it/s]
Validation DataLoader 0:  45%|████▍     | 17/38 [00:00<00:00, 23.05it/s]
Validation DataLoader 0:  47%|████▋     | 18/38 [00:00<00:00, 23.58it/s]
Validation DataLoader 0:  50%|█████     | 19/38 [00:00<00:00, 23.98it/s]
Validation DataLoader 0:  53%|█████▎    | 20/38 [00:00<00:00, 24.46it/s]
Validation DataLoader 0:  55%|█████▌    | 21/38 [00:00<00:00, 24.86it/s]
Validation DataLoader 0:  58%|█████▊    | 22/38 [00:00<00:00, 25.30it/s]
Validation DataLoader 0:  61%|██████    | 23/38 [00:00<00:00, 25.68it/s]
Validation DataLoader 0:  63%|██████▎   | 24/38 [00:00<00:00, 26.03it/s]
Validation DataLoader 0:  66%|██████▌   | 25/38 [00:00<00:00, 26.34it/s]
Validation DataLoader 0:  68%|██████▊   | 26/38 [00:00<00:00, 26.64it/s]
Validation DataLoader 0:  71%|███████   | 27/38 [00:01<00:00, 26.92it/s]
Validation DataLoader 0:  74%|███████▎  | 28/38 [00:01<00:00, 27.23it/s]
Validation DataLoader 0:  76%|███████▋  | 29/38 [00:01<00:00, 27.51it/s]
Validation DataLoader 0:  79%|███████▉  | 30/38 [00:01<00:00, 27.72it/s]
Validation DataLoader 0:  82%|████████▏ | 31/38 [00:01<00:00, 28.00it/s]
Validation DataLoader 0:  84%|████████▍ | 32/38 [00:01<00:00, 28.26it/s]
Validation DataLoader 0:  87%|████████▋ | 33/38 [00:01<00:00, 28.49it/s]
Validation DataLoader 0:  89%|████████▉ | 34/38 [00:01<00:00, 28.76it/s]
Validation DataLoader 0:  92%|█████████▏| 35/38 [00:01<00:00, 28.97it/s]
Validation DataLoader 0:  95%|█████████▍| 36/38 [00:01<00:00, 29.20it/s]
Validation DataLoader 0:  97%|█████████▋| 37/38 [00:01<00:00, 29.42it/s]
Epoch 0: 100%|██████████| 61/61 [00:47<00:00,  1.29it/s, v_num=0, train_loss_step=1.310, train_acc_step=0.500, val_loss=0.249, val_acc=0.888]
Epoch 1: 100%|██████████| 61/61 [00:25<00:00,  2.39it/s, v_num=0, train_loss_step=0.530, train_acc_step=0.500, val_loss=0.249, val_acc=0.888, train_loss_epoch=0.544, train_acc_epoch=0.705]
```
模型默认保存路径
![image.png](https://cdn.nlark.com/yuque/0/2023/png/32590345/1683875397639-6c6f35bf-70e6-41de-b524-350409a946da.png#averageHue=%233d4248&clientId=ud2825136-9429-4&from=paste&height=432&id=uafa819c9&originHeight=432&originWidth=467&originalType=binary&ratio=1&rotation=0&showTitle=false&size=22148&status=done&style=none&taskId=u53d24fca-a2b6-49f5-a213-7be81cbf0a3&title=&width=467)
### 4.3 测试

1. 导入必要的包，包括 numpy、torch、matplotlib 和自定义的 exp_dataset 和 exp_model 模块。
2. 定义 test 函数，其参数包括数据集路径 data_dir 和模型路径 model_dir。
3. 调用 get_dataloaders 函数加载数据集，并获取数据载入器。然后将类别名称保存至变量 class_names 中。
4. 加载 ResNet 模型，load_from_checkpoint 方法会从文件中加载已经训练好的模型的参数，然后将模型冻结来保持不变。
5. 将 ResNet 模型中的所有参数冻结。
6. 定义设备，如果有 GPU 的话，则使用 GPU （cuda）。将 ResNet 模型移动到设备上。
7. 迭代验证数据加载器，每次处理一个 batch。将数据转换为 tensor 并将其移动到设备上。调用模型进行推理，将输出 output 保存起来并将其最大值的索引作为预测结果 preds。
8. 循环每个 batch 中的每张图片，将其从 tensor 转换为 numpy.ndarray，并进行预处理。最后使用 Matplotlib 在一个网格中绘制图片，并为每张图片添加预测类别的标签。
9. 在 main 函数中调用 test 函数，传入数据集路径和模型路径。
```python
import numpy as np
import torch
from matplotlib import pyplot as plt

from exp_dataset import get_dataloaders
from exp_model import ResNet


def test(data_dir, model_dir):
    # 获取数据集
    datasets, dataloaders = get_dataloaders(data_dir=data_dir)
    class_names = datasets['train'].classes
    # resnet
    resnet = ResNet.load_from_checkpoint(model_dir)
    # 冻结所有参数
    resnet.freeze()

    # 没有测试集，用验证集凑合以下
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = resnet.to(device)
    for i, (inputs, labels) in enumerate(dataloaders['val']):
        inputs = inputs.to(device)
        outputs = resnet(inputs)
        _, preds = torch.max(outputs, 1)

        for j in range(inputs.size()[0]):
            # 转换 tensor to image array
            inp = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            inp = std * inp + mean
            inp = np.clip(inp, 0, 1)

            # 绘制图像网格
            ax = plt.subplot(inputs.size()[0] // 2, 2, j)
            ax.axis('off')
            ax.set_title(f'predicted: {class_names[preds[j]]}')
            ax.imshow(inp)
        plt.show()


if __name__ == "__main__":
    test(data_dir='data/hymenoptera_data',
         model_dir='lightning_logs/version_0/checkpoints/epoch=14-val_loss=0.21.ckpt')

```
结果
![image.png](https://cdn.nlark.com/yuque/0/2023/png/32590345/1683877000546-7c6cf506-cae1-4e8e-82ee-9f8973e68fa2.png#averageHue=%23d3d3bf&clientId=ud2825136-9429-4&from=paste&height=555&id=u84cd83c5&originHeight=555&originWidth=656&originalType=binary&ratio=1&rotation=0&showTitle=false&size=245683&status=done&style=none&taskId=u256b2d1a-71ed-4ba0-994d-5162cfc373d&title=&width=656)
![image.png](https://cdn.nlark.com/yuque/0/2023/png/32590345/1683877040521-91583d84-01d4-443d-a805-6e52975e9492.png#averageHue=%23dfdeca&clientId=ud2825136-9429-4&from=paste&height=480&id=ue687cb79&originHeight=480&originWidth=640&originalType=binary&ratio=1&rotation=0&showTitle=false&size=261295&status=done&style=none&taskId=u7a510ace-8576-40ef-b02d-b9db962fd67&title=&width=640)
## 5、只训练分类器
这是另一种训练方式
将预训练模型中除分类器以外的部分作为一个特征提取器（不训练参数）
添加一个分类器，分类提取到的特征即可（训练参数）
### 5.1 模型
使用 pytorch lighting的框架定义模型，简而言之使用的 pytorch lighting实现 pl.LightningModule 父类的函数，在调用时pytorch lighting 框架会自动调用 training_step、validation_step 等，而且不需要写对象传递cuda、梯度、loss回传等步骤，以下步骤会被自动执行
你可以在 [https://lightning.ai/docs/pytorch/stable/starter/introduction.html](https://lightning.ai/docs/pytorch/stable/starter/introduction.html) 快速了解它
```python
# put model in train mode and enable gradient calculation
model.train()
torch.set_grad_enabled(True)
for batch_idx, batch in enumerate(train_dataloader):
    loss = training_step(batch, batch_idx)
    # clear gradients
    optimizer.zero_grad()
    # backward
    loss.backward()
    # update parameters
    optimizer.step()
```

1. 导入需要的 PyTorch Lightning、PyTorch 和 torchvision 库。
2. 创建一个 PyTorch Lightning 的 LightningModule 类 ResNet。其中，加载了预训练的 ResNet101 模型，获取分类器的输入特征维度，替换了分类器为一个全连接层，并定义使用交叉熵作为 loss 计算方式。
3. 实现 forward、training_step、validation_step 和 configure_optimizers 四个函数。其中，forward 函数将输入数据传入 ResNet 模型并返回输出结果。training_step 和 validation_step 分别实现了训练和验证每个 batch 的具体操作，包括输入、调用 forward 函数、计算 loss 和准确率等，将计算结果通过 log_dict 函数记录在 tensorboard 日志文件中。configure_optimizers 定义了使用 SGD 优化器和每7轮学习率乘以 0.1 的学习率调节器训练模型。
```python
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import models


class FeatureExtractor(pl.LightningModule):
    def __init__(self, num_classes=2):
        super().__init__()
        # 加载resnet101的预训练模型，并将除分类器以外作为特征提取器
        # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet101.html#torchvision.models.ResNet101_Weights
        backbone = models.resnet50(weights="IMAGENET1K_V1")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # 设置自己的分类器
        self.classifier = nn.Linear(num_filters, num_classes)

        # loss 计算方式，交叉熵
        self.criterion = nn.CrossEntropyLoss()

        # pytroch_lighting 的 api 会计算 forword 的 summary
        self.example_input_array = torch.Tensor(4, 3, 224, 224)

    def forward(self, x):
        # 固定特征提取器的参数，不参加训练
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        # 训练分类器
        x = self.classifier(representations)
        return x

    def training_step(self, batch, batch_idx):
        # 获取 input 和 label
        (inputs, labels) = batch
        # 调用 forword 函数，预测
        outputs = self(inputs)
        # 第一结果是最大的值，第二个结果是最大的下标即预测的结果
        _, preds = torch.max(outputs, dim=1)
        # 计算 loss
        loss = self.criterion(outputs, labels)
        acc = torch.sum(preds == labels.data) * 1.0 / inputs.size()[0]
        # log train_loss
        self.log_dict({"train_loss": loss, "train_acc": acc}, prog_bar=True, logger=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # 获取 input 和 label
        (inputs, labels) = batch
        # 调用 forword 函数，预测
        outputs = self(inputs)
        # 第一结果是最大的值，第二个结果是最大的下标即预测的结果
        _, preds = torch.max(outputs, dim=1)
        # 计算 loss 和 acc
        loss = self.criterion(outputs, labels)
        acc = torch.sum(preds == labels.data) * 1.0 / inputs.size()[0]
        # log train_loss
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        # 分类器参数优化
        optimizer_ft = optim.SGD(self.classifier.parameters(), lr=0.001, momentum=0.9)
        # 每7轮学习率乘0.1
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        return [optimizer_ft], [exp_lr_scheduler]
```
### 5.2 训练

1. 调用了两个自定义函数：get_dataloaders和ResNet来构建数据加载器和模型。
2. 在训练过程中，它使用ModelCheckpoint回调函数来保存验证loss最低的模型。 
3. 训练过程使用单个GPU进行加速，并使用32位精度进行计算。
4. 最后，训练函数被调用并传入数据集目录作为参数。
```python
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from exp2_model import FeatureExtractor
from exp_dataset import get_dataloaders


def train(data_dir):
    # 获取数据集
    datasets, dataloaders = get_dataloaders(data_dir=data_dir)

    # resnet
    resnet = FeatureExtractor()
    # checkpoint 保存策略, 保存 验证acc最大的模型，且只保存1个
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode='min', filename='{epoch}-{val_loss:.2f}', save_top_k=1)
    # 使用第0张gpu训练，精度为32位
    trainer = pl.Trainer(accelerator="gpu", devices=[0], precision=32, max_epochs=25, callbacks=[checkpoint_callback])
    # 使用model，训练集和验证集开始训练
    trainer.fit(model=resnet,
                train_dataloaders=dataloaders['train'],
                val_dataloaders=dataloaders['val'])


if __name__ == "__main__":
    train(data_dir='data/hymenoptera_data')
```
训练过程
```python

  | Name              | Type             | Params | In sizes         | Out sizes
--------------------------------------------------------------------------------------------
0 | feature_extractor | Sequential       | 23.5 M | [4, 3, 224, 224] | [4, 2048, 1, 1]
1 | classifier        | Linear           | 4.1 K  | [4, 2048]        | [4, 2]
2 | criterion         | CrossEntropyLoss | 0      | ?                | ?
--------------------------------------------------------------------------------------------
23.5 M    Trainable params
0         Non-trainable params
23.5 M    Total params
94.049    Total estimated model params size (MB)
Epoch 24: 100%|███████████████████████████| 61/61 [00:25<00:00,  2.36it/s, v_num=1, train_loss_step=0.012, train_acc_step=1.000, val_loss=0.0961, val_acc=0.967, train_loss_epoch=0.0626, train_acc_epoch=0.984]`Trainer.fit` stopped: `max_epochs=25` reached.
Epoch 24: 100%|███████████████████████████| 61/61 [00:25<00:00,  2.36it/s, v_num=1, train_loss_step=0.012, train_acc_step=1.000, val_loss=0.0961, val_acc=0.967, train_loss_epoch=0.0626, train_acc_epoch=0.984]
```
### 5.3 测试

1. 导入必要的包，包括 numpy、torch、matplotlib 和自定义的 exp_dataset 和 exp_model 模块。
2. 定义 test 函数，其参数包括数据集路径 data_dir 和模型路径 model_dir。
3. 调用 get_dataloaders 函数加载数据集，并获取数据载入器。然后将类别名称保存至变量 class_names 中。
4. 加载 ResNet 模型，load_from_checkpoint 方法会从文件中加载已经训练好的模型的参数，然后将模型冻结来保持不变。
5. 将 ResNet 模型中的所有参数冻结。
6. 定义设备，如果有 GPU 的话，则使用 GPU （cuda）。将 ResNet 模型移动到设备上。
7. 迭代验证数据加载器，每次处理一个 batch。将数据转换为 tensor 并将其移动到设备上。调用模型进行推理，将输出 output 保存起来并将其最大值的索引作为预测结果 preds。
8. 循环每个 batch 中的每张图片，将其从 tensor 转换为 numpy.ndarray，并进行预处理。最后使用 Matplotlib 在一个网格中绘制图片，并为每张图片添加预测类别的标签。
9. 在 main 函数中调用 test 函数，传入数据集路径和模型路径。
```python
import numpy as np
import torch
from matplotlib import pyplot as plt

from exp2_model import FeatureExtractor
from exp_dataset import get_dataloaders


def test(data_dir, model_dir):
    # 获取数据集
    datasets, dataloaders = get_dataloaders(data_dir=data_dir)
    class_names = datasets['train'].classes
    # resnet
    resnet = FeatureExtractor.load_from_checkpoint(model_dir)
    # 冻结所有参数
    resnet.freeze()

    # 没有测试集，用验证集凑合以下
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = resnet.to(device)
    for i, (inputs, labels) in enumerate(dataloaders['val']):
        inputs = inputs.to(device)
        outputs = resnet(inputs)
        _, preds = torch.max(outputs, 1)

        for j in range(inputs.size()[0]):
            # 转换 tensor to image array
            inp = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            inp = std * inp + mean
            inp = np.clip(inp, 0, 1)

            # 绘制图像网格
            ax = plt.subplot(inputs.size()[0] // 2, 2, j+1)
            ax.axis('off')
            ax.set_title(f'predicted: {class_names[preds[j]]}')
            ax.imshow(inp)
        plt.show()


if __name__ == "__main__":
    test(data_dir='data/hymenoptera_data',
         model_dir='lightning_logs/version_0/checkpoints/epoch=14-val_loss=0.21.ckpt')

```
![image.png](https://cdn.nlark.com/yuque/0/2023/png/32590345/1683879313400-b5a7b387-b990-4a62-a5dd-3661f686c744.png#averageHue=%23d8d3cd&clientId=ud2825136-9429-4&from=paste&height=480&id=uee0a0f86&originHeight=480&originWidth=640&originalType=binary&ratio=1&rotation=0&showTitle=false&size=269666&status=done&style=none&taskId=u3575683d-f80f-4f6f-8ed0-a3f43e433ae&title=&width=640)
![image.png](https://cdn.nlark.com/yuque/0/2023/png/32590345/1683879338041-9a4cf8dc-b24d-4d6f-9f2b-7ad088c6b6cb.png#averageHue=%23686348&clientId=ud2825136-9429-4&from=paste&height=480&id=u2b6e297f&originHeight=480&originWidth=640&originalType=binary&ratio=1&rotation=0&showTitle=false&size=257576&status=done&style=none&taskId=ucddaab17-5db3-42f4-9b49-94db1c64caa&title=&width=640)
# 自己的实验
## 1、自己的数据集
将模型应用于自己数据集，需要自己重写数据集，即需要重写 torch.utils.data.Dataset类，并重写 def __len__(self) 和 def __getitem__(self, idx) 方法，分别实现返回数据集大小和获取数据集的项的方法
为了方便，我简陋实现了类似ImageFolder的代码，可以改写下面的代码来实现自己的数据集
```python
import os

import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms, datasets


class ClassificationDataset(Dataset):
    def __init__(self, data_dir, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])):
        # 文件路径
        self.data_dir = data_dir
        # 分类集
        self.classes = os.listdir(self.data_dir)
        # 双重循环构造data list，每个item由(图片路径，label)构成
        self.data = [
            (os.path.join(self.data_dir, path_name, file), label)
            for label, path_name in enumerate(self.classes)
            for file in os.listdir(os.path.join(self.data_dir, path_name))]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (image_path, label) = self.data[idx]
        image = Image.open(image_path)
        image = self.transform(image)
        return image, label


# 获取数据集
def get_dataloaders(data_dir):
    # 定义数据路径
    data_dir = 'data/hymenoptera_data'

    # 定义图像的变换和数据增强 transform
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 使用 ImageFolder 类自动构造分类数据集，会将文件夹名作为分类标签
    image_datasets = {x: ClassificationDataset(os.path.join(data_dir, x),
                                                        data_transforms[x])
                      for x in ['train', 'val']}
    # 构造 dataloader
    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=4,
                                                        shuffle=True, num_workers=4, drop_last=True),
                   'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=4, num_workers=4,
                                                      drop_last=True),
                   }
    return image_datasets, dataloaders


# 从tensor 转化为图片并显示
def imshow(inp, title=None):
    # 从 chw -> hwc
    inp = inp.numpy().transpose((1, 2, 0))
    # 还原归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    # 显示图像
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(20)


if __name__ == "__main__":
    image_datasets, dataloaders = get_dataloaders(data_dir='data/hymenoptera_data')
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # 获取类的名称集合
    class_names = image_datasets['train'].classes

    # 查看数据集
    inputs, classes = next(iter(dataloaders['train']))
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])

```
## 2、自己的模型
这是实验中的模型代码，你可以直接修改，或者直接用

1. 初始化模型的 num_classes，来修改分类种类
2.  models.resnet101(weights='IMAGENET1K_V1') 修改weights来选择初始化的预训练模型 [https://pytorch.org/vision/main/models/generated/torchvision.models.resnet101.html#torchvision.models.ResNet101_Weights](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet101.html#torchvision.models.ResNet101_Weights)
3. self.model.fc = nn.Linear(num_ftrs, num_classes) 修改分类器
4. self.criterion = nn.CrossEntropyLoss() 修改常用的loss计算方式
5. optimizer_ft = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)、exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)修改优化器和学习率等参数
6. 训练整个模型时，也可以锁住前几层，只训练后面几层
```python
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import models


# define the LightningModule
class ResNet(pl.LightningModule):
    def __init__(self, num_classes=2):
        super().__init__()
        # 加载resnet101的预训练模型
        # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet101.html#torchvision.models.ResNet101_Weights
        self.model = models.resnet101(weights='IMAGENET1K_V1')
        # 获取分类器的输入特征维度
        num_ftrs = self.model.fc.in_features
        # 替换分类器
        self.model.fc = nn.Linear(num_ftrs, num_classes)

        # loss 计算方式，交叉熵
        self.criterion = nn.CrossEntropyLoss()

        # pytroch_lighting 的 api 会计算 forword 的 summary
        self.example_input_array = torch.Tensor(4, 3, 224, 224)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # 获取 input 和 label
        (inputs, labels) = batch
        # 调用 forword 函数，预测
        outputs = self(inputs)
        # 第一结果是最大的值，第二个结果是最大的下标即预测的结果
        _, preds = torch.max(outputs, dim=1)
        # 计算 loss
        loss = self.criterion(outputs, labels)
        acc = torch.sum(preds == labels.data) * 1.0 / inputs.size()[0]
        # log train_loss
        self.log_dict({"train_loss": loss, "train_acc": acc}, prog_bar=True, logger=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # 获取 input 和 label
        (inputs, labels) = batch
        # 调用 forword 函数，预测
        outputs = self(inputs)
        # 第一结果是最大的值，第二个结果是最大的下标即预测的结果
        _, preds = torch.max(outputs, dim=1)
        # 计算 loss 和 acc
        loss = self.criterion(outputs, labels)
        acc = torch.sum(preds == labels.data) * 1.0 / inputs.size()[0]
        # log train_loss
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        # 所有参数优化
        optimizer_ft = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        # 每7轮学习率乘0.1
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        return [optimizer_ft], [exp_lr_scheduler]

class FeatureExtractor(pl.LightningModule):
    def __init__(self, num_classes=2):
        super().__init__()
        # 加载resnet101的预训练模型，并将除分类器以外作为特征提取器
        # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet101.html#torchvision.models.ResNet101_Weights
        backbone = models.resnet50(weights="IMAGENET1K_V1")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # 设置自己的分类器
        self.classifier = nn.Linear(num_filters, num_classes)

        # loss 计算方式，交叉熵
        self.criterion = nn.CrossEntropyLoss()

        # pytroch_lighting 的 api 会计算 forword 的 summary
        self.example_input_array = torch.Tensor(4, 3, 224, 224)

    def forward(self, x):
        # 固定特征提取器的参数，不参加训练
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        # 训练分类器
        x = self.classifier(representations)
        return x

    def training_step(self, batch, batch_idx):
        # 获取 input 和 label
        (inputs, labels) = batch
        # 调用 forword 函数，预测
        outputs = self(inputs)
        # 第一结果是最大的值，第二个结果是最大的下标即预测的结果
        _, preds = torch.max(outputs, dim=1)
        # 计算 loss
        loss = self.criterion(outputs, labels)
        acc = torch.sum(preds == labels.data) * 1.0 / inputs.size()[0]
        # log train_loss
        self.log_dict({"train_loss": loss, "train_acc": acc}, prog_bar=True, logger=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # 获取 input 和 label
        (inputs, labels) = batch
        # 调用 forword 函数，预测
        outputs = self(inputs)
        # 第一结果是最大的值，第二个结果是最大的下标即预测的结果
        _, preds = torch.max(outputs, dim=1)
        # 计算 loss 和 acc
        loss = self.criterion(outputs, labels)
        acc = torch.sum(preds == labels.data) * 1.0 / inputs.size()[0]
        # log train_loss
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        # 分类器参数优化
        optimizer_ft = optim.SGD(self.classifier.parameters(), lr=0.001, momentum=0.9)
        # 每7轮学习率乘0.1
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        return [optimizer_ft], [exp_lr_scheduler]
```
## 3、训练和测试
同上
```python
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from exp_dataset import get_dataloaders
from exp_model import ResNet


def train(data_dir):
    # 获取数据集
    datasets, dataloaders = get_dataloaders(data_dir=data_dir)

    # resnet
    resnet = ResNet()
    # checkpoint 保存策略, 保存 验证acc最大的模型，且只保存1个
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode='min', filename='{epoch}-{val_loss:.2f}', save_top_k=1)
    # 使用第0张gpu训练，精度为32位
    trainer = pl.Trainer(accelerator="gpu", devices=[0], precision=32, max_epochs=25, callbacks=[checkpoint_callback])
    # 使用model，训练集和验证集开始训练
    trainer.fit(model=resnet,
                train_dataloaders=dataloaders['train'],
                val_dataloaders=dataloaders['val'])


if __name__ == "__main__":
    train(data_dir='data/hymenoptera_data')
```
同上
```python
import numpy as np
import torch
from matplotlib import pyplot as plt

from exp_dataset import get_dataloaders
from exp_model import ResNet


def test(data_dir, model_dir):
    # 获取数据集
    datasets, dataloaders = get_dataloaders(data_dir=data_dir)
    class_names = datasets['train'].classes
    # resnet
    resnet = ResNet.load_from_checkpoint(model_dir)
    # 冻结所有参数
    resnet.freeze()

    # 没有测试集，用验证集凑合以下
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = resnet.to(device)
    for i, (inputs, labels) in enumerate(dataloaders['val']):
        inputs = inputs.to(device)
        outputs = resnet(inputs)
        _, preds = torch.max(outputs, 1)

        for j in range(inputs.size()[0]):
            # 转换 tensor to image array
            inp = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            inp = std * inp + mean
            inp = np.clip(inp, 0, 1)

            # 绘制图像网格
            ax = plt.subplot(inputs.size()[0] // 2, 2, j+1)
            ax.axis('off')
            ax.set_title(f'predicted: {class_names[preds[j]]}')
            ax.imshow(inp)
        plt.show()


if __name__ == "__main__":
    test(data_dir='data/hymenoptera_data',
         model_dir='lightning_logs/version_0/checkpoints/epoch=14-val_loss=0.21.ckpt')
```
