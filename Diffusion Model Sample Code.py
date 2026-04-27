import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 超参数设置
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
TIMESTEPS = 1000

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 定义噪声调度
def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


# 预计算扩散过程所需的各种值
betas = linear_beta_schedule(TIMESTEPS).to(device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


# 简化的UNet模型
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()

        # 编码器
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2)

        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
        )

        # 解码器
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),  # 128 + 128 from enc2
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),  # 64 + 64 from enc1
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),  # 输出通道为1
        )

        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

    def forward(self, x, t):
        # 时间嵌入并添加到瓶颈层
        t_embed = self.time_embed(t.unsqueeze(-1).float())
        t_embed = t_embed.view(-1, 256, 1, 1)

        # 编码器路径
        enc1 = self.enc1(x)  # 64x28x28
        enc1_pool = self.pool1(enc1)  # 64x14x14

        enc2 = self.enc2(enc1_pool)  # 128x14x14
        enc2_pool = self.pool2(enc2)  # 128x7x7

        # 瓶颈层 + 时间嵌入
        bottleneck = self.bottleneck(enc2_pool)  # 256x7x7
        bottleneck = bottleneck + t_embed

        # 解码器路径
        up2 = self.up2(bottleneck)  # 128x14x14
        # 确保尺寸匹配
        if up2.shape != enc2.shape:
            up2 = F.interpolate(up2, size=enc2.shape[2:], mode='nearest')
        dec2_input = torch.cat([up2, enc2], dim=1)  # 256x14x14
        dec2 = self.dec2(dec2_input)  # 128x14x14

        up1 = self.up1(dec2)  # 64x28x28
        # 确保尺寸匹配
        if up1.shape != enc1.shape:
            up1 = F.interpolate(up1, size=enc1.shape[2:], mode='nearest')
        dec1_input = torch.cat([up1, enc1], dim=1)  # 128x28x28
        output = self.dec1(dec1_input)  # 1x28x28

        return output


# 初始化模型
model = SimpleUNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# 定义前向扩散过程（添加噪声）
def forward_diffusion(x0, t, device):
    noise = torch.randn_like(x0)
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

    # 添加噪声
    xt = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
    return xt, noise


# 训练函数
def train_model():
    model.train()
    losses = []

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0

        for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")):
            data = data.to(device)
            batch_size = data.shape[0]

            # 随机选择时间步
            t = torch.randint(0, TIMESTEPS, (batch_size,), device=device).long()

            # 前向扩散过程
            noisy_data, noise = forward_diffusion(data, t, device)

            # 预测噪声
            predicted_noise = model(noisy_data, t)

            # 计算损失
            loss = F.mse_loss(predicted_noise, noise)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

        # 每10个epoch保存一次模型
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'mnist_diffusion_model_epoch_{epoch + 1}.pth')
            print(f"模型已保存为 'mnist_diffusion_model_epoch_{epoch + 1}.pth'")

    return losses


# 定义采样函数（反向扩散过程）
@torch.no_grad()
def sample(model, image_size=28, batch_size=16, channels=1):
    model.eval()

    # 从纯噪声开始
    img = torch.randn((batch_size, channels, image_size, image_size), device=device)

    for i in tqdm(reversed(range(0, TIMESTEPS)), desc="Sampling"):
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)

        # 预测噪声
        predicted_noise = model(img, t)

        # 计算系数
        alpha = alphas[t].view(-1, 1, 1, 1)
        alpha_cumprod = alphas_cumprod[t].view(-1, 1, 1, 1)
        beta = betas[t].view(-1, 1, 1, 1)

        if i > 0:
            noise = torch.randn_like(img)
        else:
            noise = torch.zeros_like(img)

        # 反向扩散步骤
        img = 1 / torch.sqrt(alpha) * (
                    img - ((1 - alpha) / (torch.sqrt(1 - alpha_cumprod))) * predicted_noise) + torch.sqrt(beta) * noise

    # 将图像缩放到 [0, 1] 范围
    img = (img + 1) / 2
    img = img.clamp(0, 1)

    return img


# 加载预训练模型并生成图像的函数
def load_and_generate(model_path, num_images=16, save_path='generated_images.png'):
    """
    加载预训练模型并生成图像

    参数:
        model_path: 模型文件路径
        num_images: 要生成的图像数量
        save_path: 生成图像的保存路径
    """
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 '{model_path}' 不存在!")
        return

    # 初始化模型
    loaded_model = SimpleUNet().to(device)

    try:
        # 加载模型权重
        loaded_model.load_state_dict(torch.load(model_path, map_location=device))
        loaded_model.eval()
        print(f"成功加载模型: {model_path}")

        # 生成图像
        print(f"正在生成 {num_images} 张数字图像...")
        generated_images = sample(loaded_model, batch_size=num_images)

        # 计算网格尺寸
        grid_size = int(np.ceil(np.sqrt(num_images)))

        # 显示生成的图像
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

        for i, ax in enumerate(axes.flat):
            if i < num_images:
                ax.imshow(generated_images[i].cpu().squeeze(), cmap='gray')
            ax.axis('off')

        plt.tight_layout()

        # 保存图像
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"生成的图像已保存为: {save_path}")
        plt.show()

        return generated_images

    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None


# 主程序
if __name__ == "__main__" :
    import argparse

    parser = argparse.ArgumentParser(description='MNIST扩散模型训练和生成')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'generate'],
                        help='运行模式: train (训练) 或 generate (生成)')
    parser.add_argument('--model_path', type=str, default='mnist_diffusion_model.pth',
                        help='模型文件路径 (用于生成模式)')
    parser.add_argument('--num_images', type=int, default=16,
                        help='要生成的图像数量 (用于生成模式)')
    parser.add_argument('--output_path', type=str, default='generated_digits.png',
                        help='生成图像的保存路径 (用于生成模式)')

    args = parser.parse_args()

    if args.mode == 'train' and False:
        print("开始训练扩散模型...")
        losses = train_model()

        # 绘制损失曲线
        plt.plot(losses)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
        plt.show()

        # 保存最终模型
        torch.save(model.state_dict(), 'mnist_diffusion_model_final.pth')
        print("最终模型已保存为 'mnist_diffusion_model_final.pth'")

        # 训练后立即生成示例图像
        print("使用训练好的模型生成示例图像...")
        generated_images = sample(model, batch_size=16)

        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            ax.imshow(generated_images[i].cpu().squeeze(), cmap='gray')
            ax.axis('off')
        plt.tight_layout()
        plt.savefig('final_generated_digits.png', dpi=150, bbox_inches='tight')
        plt.show()

    else:
        # 直接使用保存的模型生成图像
        load_and_generate(
            model_path=args.model_path,
            num_images=args.num_images,
            save_path=args.output_path
        )