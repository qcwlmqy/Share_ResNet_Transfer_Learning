import numpy as np
import torch
from matplotlib import pyplot as plt

from own_dataset import get_dataloaders
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
