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
