import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import argparse

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据预处理（训练和推理使用相同的预处理）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# 定义神经网络模型
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


# 训练函数
def train_model(model_path='mnist_model.pth', epochs=5, resume_training=False):
    """训练MNIST分类模型"""

    # 加载数据集
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 初始化模型
    model = NeuralNet().to(device)
    start_epoch = 0

    # 如果选择继续训练且模型文件存在，则加载模型
    if resume_training and os.path.exists(model_path):
        print(f"加载已有模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"从第 {start_epoch} 轮开始继续训练")
    else:
        print("从头开始训练新模型")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 如果继续训练，加载优化器状态
    if resume_training and os.path.exists(model_path):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 训练循环
    model.train()
    train_losses = []
    train_accuracies = []

    for epoch in range(start_epoch, start_epoch + epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # 前向传播
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # 反向传播
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if batch_idx % 100 == 0:
                print(
                    f'Epoch: {epoch + 1}/{start_epoch + epochs} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100. * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        print(f'Epoch {epoch + 1}完成 - 损失: {epoch_loss:.4f}, 准确率: {epoch_accuracy:.2f}%')

        # 每轮结束后保存检查点
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            'accuracy': epoch_accuracy
        }, model_path)
        print(f"模型已保存: {model_path}")

    # 最终测试
    test_accuracy = test_model(model, test_loader)
    print(f'最终测试集准确率: {test_accuracy:.2f}%')

    # 可视化训练过程
    if epochs > 0:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title('训练损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies)
        plt.title('训练准确率')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()

    return model


# 测试函数
def test_model(model, test_loader):
    """测试模型准确率"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = 100. * correct / total
    return accuracy


# 图像预处理函数（用于推理）
def preprocess_image(image_path):
    """预处理单张图像用于推理"""
    image = Image.open(image_path).convert('L')  # 转换为灰度图
    image = image.resize((28, 28))  # 调整大小为28x28

    # 转换为numpy数组并反转颜色（MNIST是白字黑底）
    image_array = np.array(image)
    image_array = 255 - image_array  # 反转颜色

    # 转换为tensor并应用相同的预处理
    image_tensor = transform(image_array)
    image_tensor = image_tensor.unsqueeze(0)  # 添加batch维度

    return image_tensor


# 推理函数
def predict_images(model, image_folder, model_path='mnist_model.pth',tag=7):
    """对文件夹中的图像进行分类预测"""

    # 加载模型
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在！")
        return

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"模型加载成功: {model_path}")

    # 获取图像文件列表
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_files = []

    for file in os.listdir(image_folder):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(image_folder, file))

    if not image_files:
        print(f"在文件夹 {image_folder} 中未找到图像文件")
        return

    print(f"找到 {len(image_files)} 张图像")

    # 对每张图像进行预测
    results = []
    cnt=0

    for image_path in image_files:
        try:
            # 预处理图像
            image_tensor = preprocess_image(image_path)
            image_tensor = image_tensor.to(device)

            # 预测
            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                predicted_class = output.argmax(dim=1).item()
                confidence = probabilities[predicted_class].item()

            results.append({
                'image_path': image_path,
                'predicted_class': predicted_class,
                'confidence': confidence
            })
            cnt+=int(predicted_class==tag)
            #print(f"图像: {os.path.basename(image_path)} -> 预测数字: {predicted_class}, 置信度: {confidence:.4f}")

            # 显示图像和预测结果
            plt.figure(figsize=(4, 4))
            image = Image.open(image_path).convert('L')
            plt.imshow(image, cmap='gray')
            plt.title(f'预测: {predicted_class} (置信度: {confidence:.4f})')
            plt.axis('off')
            plt.tight_layout()
            #plt.show()

        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {e}")
    print(cnt/len(image_files))
    return results


# 主函数
def main():
    parser = argparse.ArgumentParser(description='MNIST分类训练和推理')
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], required=True,
                        help='模式: train(训练) 或 predict(推理)')
    parser.add_argument('--resume', type=bool, default=False,
                        help='是否继续训练 (仅训练模式有效)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='训练轮数 (仅训练模式有效)')
    parser.add_argument('--model_path', type=str, default='mnist_model.pth',
                        help='模型文件路径')
    parser.add_argument('--image_folder', type=str, default='./images',
                        help='包含待分类图像的文件夹路径 (仅推理模式有效)')

    args = parser.parse_args()

    # 初始化模型
    model = NeuralNet().to(device)

    if args.mode == 'train':
        print("=== 训练模式 ===")
        train_model(
            model_path=args.model_path,
            epochs=args.epochs,
            resume_training=args.resume
        )

    elif args.mode == 'predict':
        print("=== 推理模式 ===")
        if not os.path.exists(args.image_folder):
            print(f"错误: 图像文件夹 {args.image_folder} 不存在！")
            return

        predict_images(
            model=model,
            image_folder=args.image_folder,
            model_path=args.model_path
        )


# 简化使用函数（无需命令行参数）
def simple_usage(is_training=True, image_folder='./images', epochs=5, resume=False):
    """
    简化使用函数

    参数:
    - is_training: True表示训练模式，False表示推理模式
    - image_folder: 推理时使用的图像文件夹路径
    - epochs: 训练轮数
    - resume: 是否继续训练
    """
    model = NeuralNet().to(device)

    if is_training:
        print("=== 训练模式 ===")
        train_model(epochs=epochs, resume_training=resume)
    else:
        print("=== 推理模式 ===")
        if not os.path.exists(image_folder):
            print(f"创建示例图像文件夹: {image_folder}")
            os.makedirs(image_folder, exist_ok=True)
            print(f"请将待分类的MNIST风格数字图像放入 {image_folder} 文件夹中")
            return

        predict_images(model=model, image_folder=image_folder)


if __name__ == "__main__":

    simple_usage(is_training=False, image_folder='./experiment_results_20250924_183837/sinkhorn_samples/sinkhorn_model')

