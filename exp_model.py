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
