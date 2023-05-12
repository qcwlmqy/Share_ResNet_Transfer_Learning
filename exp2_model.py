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
