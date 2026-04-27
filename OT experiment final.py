import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.utils import make_grid
# --- 1. 超参数设置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 10  # [优化] 减小批大小，以适应50个样本 (50/10 = 5 个批次)
num_epochs = 500  # [优化] 大幅增加训练轮数
learning_rate = 2e-4  # [优化] 调整学习率
T = 400  # [优化] 增加时间步长
beta_start = 0.0001
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T, device=device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# Sinkhorn 参数
sinkhorn_epsilon = 0.01
sinkhorn_n_iter = 50  # [优化] 增加迭代次数以获得更稳定的OT梯度
ot_loss_weight = 0.1
ot_guidance_weight = 2.0

# 少样本参数
target_class = 5
n_shot = 500
total_samples = 50000

print(f"Device: {device}")
print(f"N-shot: {n_shot}, Target Class: {target_class}")
print(f"Epochs: {num_epochs}, Batch Size: {batch_size}, LR: {learning_rate}, T: {T}")

# --- 2. 加载并准备少样本MNIST数据 ---
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

idx = (train_dataset.targets == target_class).nonzero(as_tuple=True)[0]
# 固定随机种子以便复现
torch.manual_seed(42)
selected_idx = idx[torch.randperm(len(idx))[:n_shot]]
few_shot_dataset = Subset(train_dataset, selected_idx)
train_loader = DataLoader(few_shot_dataset, batch_size=batch_size, shuffle=True)

# 获取一些真实样本用于OT指导和可视化
real_samples_for_guidance = next(iter(DataLoader(few_shot_dataset, batch_size=16, shuffle=True)))[0].to(device)


# --- 3. 定义神经网络 [重大优化] ---

class SinusoidalPositionEmbeddings(nn.Module):
    """
    [新增] 正弦位置编码
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class UNetBlock(nn.Module):
    """
    [新增] 带有时间注入和残差连接的U-Net基础块
    """

    def __init__(self, in_channels, out_channels, time_emb_dim, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups, out_channels)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups, out_channels)
        self.act2 = nn.ReLU()

        self.time_proj = nn.Linear(time_emb_dim, out_channels)

        # 残差连接
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, t_emb):
        residual = self.residual_conv(x)

        x = self.conv1(x)
        x = self.gn1(x)
        x = self.act1(x)

        # 注入时间嵌入
        t_emb_proj = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        x = x + t_emb_proj

        x = self.conv2(x)
        x = self.gn2(x)
        x = self.act2(x)

        return x + residual


class SimpleUNet(nn.Module):
    """
    [重写] 具有跳跃连接和正确时间嵌入的U-Net
    """

    def __init__(self, time_emb_dim=32):
        super().__init__()

        # 时间嵌入
        self.time_emb = SinusoidalPositionEmbeddings(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.ReLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # 初始卷积 (输入通道为1)
        self.initial_conv = nn.Conv2d(1, 32, 3, stride=1, padding=1)

        # 编码器
        self.enc_block1 = UNetBlock(32, 64, time_emb_dim)
        self.downsample1 = nn.Conv2d(64, 64, 4, stride=2, padding=1)  # 28x28 -> 14x14

        self.enc_block2 = UNetBlock(64, 128, time_emb_dim)
        self.downsample2 = nn.Conv2d(128, 128, 4, stride=2, padding=1)  # 14x14 -> 7x7

        # 中间
        self.mid_block1 = UNetBlock(128, 128, time_emb_dim)
        self.mid_block2 = UNetBlock(128, 128, time_emb_dim)

        # 解码器
        self.upsample1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)  # 7x7 -> 14x14
        # [优化] 跳跃连接: 64 (来自上采样) + 128 (来自 enc_block2)
        self.dec_block1 = UNetBlock(128 + 64, 64, time_emb_dim)

        self.upsample2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)  # 14x14 -> 28x28
        # [优化] 跳跃连接: 32 (来自上采样) + 64 (来自 enc_block1)
        self.dec_block2 = UNetBlock(64 + 32, 32, time_emb_dim)

        self.final_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x, t):
        # 1. 处理时间
        t_emb = self.time_emb(t)
        t_emb = self.time_mlp(t_emb)

        # 2. 编码器
        x0 = self.initial_conv(x)  # (B, 32, 28, 28)

        x1 = self.enc_block1(x0, t_emb)  # (B, 64, 28, 28)
        x1_down = self.downsample1(x1)  # (B, 64, 14, 14)

        x2 = self.enc_block2(x1_down, t_emb)  # (B, 128, 14, 14)
        x2_down = self.downsample2(x2)  # (B, 128, 7, 7)

        # 3. 中间
        xm = self.mid_block1(x2_down, t_emb)  # (B, 128, 7, 7)
        xm = self.mid_block2(xm, t_emb)  # (B, 128, 7, 7)

        # 4. 解码器 (带跳跃连接)
        x_up1 = self.upsample1(xm)  # (B, 64, 14, 14)
        x_dec1 = torch.cat([x_up1, x2], dim=1)  # (B, 64 + 128, 14, 14)
        x_dec1 = self.dec_block1(x_dec1, t_emb)  # (B, 64, 14, 14)

        x_up2 = self.upsample2(x_dec1)  # (B, 32, 28, 28)
        x_dec2 = torch.cat([x_up2, x1], dim=1)  # (B, 32 + 64, 28, 28)
        x_dec2 = self.dec_block2(x_dec2, t_emb)  # (B, 32, 28, 28)

        # 5. 最终
        out = self.final_conv(x_dec2)
        return out


# --- 4. 定义Sinkhorn算法 ---
def sinkhorn_loss(x_real, x_fake, epsilon=sinkhorn_epsilon, n_iters=sinkhorn_n_iter):
    n = x_real.size(0)
    m = x_fake.size(0)

    # 确保不为0，避免除零
    if n == 0 or m == 0:
        return torch.tensor(0.0, device=device)

    x_real_flat = x_real.view(n, -1)
    x_fake_flat = x_fake.view(m, -1)

    C = torch.cdist(x_real_flat, x_fake_flat, p=2) ** 2
    C = C / (torch.max(C).detach() + 1e-8)  # [优化] 使用 .detach()

    K = torch.exp(-C / epsilon)

    # [优化] 使用更稳定的log-space Sinkhorn
    log_K = -C / epsilon
    log_u = torch.zeros(n, device=device)
    log_v = torch.zeros(m, device=device)

    for _ in range(n_iters):
        log_u = -torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_v = -torch.logsumexp(log_K.t() + log_u.unsqueeze(0), dim=1)

    P = torch.exp(log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0))
    ot_distance = torch.sum(P * C)

    return ot_distance


# --- 5. 定义OT指导的采样函数 ---
@torch.no_grad()
def sample_with_ot_guidance(model, real_samples, num_samples=16, use_ot_guidance=False):
    model.eval()
    x = torch.randn(num_samples, 1, 28, 28, device=device)
    samples = []

    for t in tqdm(range(T - 1, -1, -1), desc="Sampling"):
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)

        # 第一次前向传播得到预测的噪声
        predicted_noise = model(x, t_batch)

        # DDPM采样公式
        alpha_t = alphas[t]
        alpha_cumprod_t = alphas_cumprod[t]
        beta_t = betas[t]

        # 计算均值
        mean = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise)

        # 如果使用OT指导，计算OT梯度并调整均值
        if use_ot_guidance and t > 0:  # 指导在t>0时均可应用
            # 需要计算关于x的梯度，所以启用梯度计算
            x_requires_grad = x.detach().requires_grad_(True)

            # 预测当前时间步的x0
            with torch.enable_grad():
                predicted_noise_grad = model(x_requires_grad, t_batch)
                predicted_x0 = (x_requires_grad - sqrt_one_minus_alphas_cumprod[t] * predicted_noise_grad) / \
                               sqrt_alphas_cumprod[t]
                predicted_x0 = torch.clamp(predicted_x0, -1.0, 1.0)

                # [优化] 确保guidance batch大小匹配
                guidance_batch_size = min(num_samples, real_samples.size(0))
                ot_loss = sinkhorn_loss(real_samples[:guidance_batch_size],
                                        predicted_x0[:guidance_batch_size])

                # 计算OT损失关于x的梯度
                ot_grad = torch.autograd.grad(ot_loss, x_requires_grad)[0]

            # [优化] 使用更标准的指导方式 (类似CFG)
            # 梯度方向是最小化OT损失，因此我们从均值中“减去”它
            # 方差项 sqrt(1-alpha_cumprod_t) 充当梯度的缩放因子
            guidance_scale = ot_guidance_weight * sqrt_one_minus_alphas_cumprod[t]
            mean = mean - guidance_scale * ot_grad

        if t > 0:
            # 计算方差并添加噪声
            variance = betas[t]
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(variance) * noise
        else:
            # 最后一步不需要添加噪声
            x = mean

        x = torch.clamp(x, -1.0, 1.0)

        if t % (T // 5) == 0 or t == 0:
            samples.append(x.cpu())

    return samples


# --- 6. 训练函数 ---
def train_model(use_ot_loss=False, model_name="model", lr=1e-3, epochs=50):
    if use_ot_loss:
        model = SimpleUNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        losses = []
        ot_losses = []
        mse_losses = []

        print(f"训练{model_name}（使用Sinkhorn损失）...")
        for epoch in range(epochs):
            model.train()
            pbar = tqdm(train_loader, desc=f'{model_name} Epoch {epoch + 1}/{epochs}')
            for i, (images, _) in enumerate(pbar):
                optimizer.zero_grad()
                images = images.to(device)
                b_size = images.size(0)
                if b_size == 0: continue

                t = torch.randint(0, T, (b_size,), device=device).long()
                noise = torch.randn_like(images)
                sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
                sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
                noisy_images = sqrt_alpha_cumprod_t * images + sqrt_one_minus_alpha_cumprod_t * noise

                predicted_noise = model(noisy_images, t)
                mse_loss = F.mse_loss(predicted_noise, noise)

                # [优化] 让OT损失在10%的训练后开始
                if epoch > epochs * 0.1:
                    with torch.no_grad():
                        predicted_x0 = (noisy_images - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / (
                                    sqrt_alpha_cumprod_t + 1e-8)
                        predicted_x0 = torch.clamp(predicted_x0, -1.0, 1.0)

                    # [优化] 不再使用[:8]子集，而是使用整个批次
                    ot_loss = sinkhorn_loss(images, predicted_x0)
                else:
                    ot_loss = torch.tensor(0.0, device=device)

                total_loss = mse_loss + ot_loss_weight * ot_loss
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                losses.append(total_loss.item())
                mse_losses.append(mse_loss.item())
                ot_losses.append(ot_loss.item() if ot_loss != 0 else 0)

                pbar.set_postfix({
                    'Total Loss': f'{total_loss.item():.4f}',
                    'MSE Loss': f'{mse_loss.item():.4f}',
                    'OT Loss': f'{ot_loss.item():.4f}' if ot_loss.item() != 0 else '0.0000'
                })

        return model, losses, mse_losses, ot_losses

    else:
        # 训练基线模型（无OT损失）
        model = SimpleUNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        losses = []

        print(f"训练{model_name}（基线，无Sinkhorn损失）...")
        for epoch in range(epochs):
            model.train()
            pbar = tqdm(train_loader, desc=f'{model_name} Epoch {epoch + 1}/{epochs}')
            for images, _ in pbar:
                images = images.to(device)
                b_size = images.size(0)
                if b_size == 0: continue

                t = torch.randint(0, T, (b_size,), device=device).long()
                noise = torch.randn_like(images)
                sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
                sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
                noisy_images = sqrt_alpha_cumprod_t * images + sqrt_one_minus_alpha_cumprod_t * noise

                predicted_noise = model(noisy_images, t)
                loss = F.mse_loss(predicted_noise, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                pbar.set_postfix({'MSE Loss': f'{loss.item():.4f}'})

        return model, losses, None, None


# --- 7. 主实验 ---
# 训练两个模型：一个使用OT损失，一个不使用
# [优化] 传入新的学习率和轮数
model_with_ot, losses_ot, mse_losses_ot, ot_losses_ot = train_model(
    use_ot_loss=True, model_name="OT模型", lr=learning_rate, epochs=num_epochs
)
model_baseline, losses_baseline, _, _ = train_model(
    use_ot_loss=False, model_name="基线模型", lr=learning_rate, epochs=num_epochs
)

# --- 8. 生成四种情况的样本 ---
print("\n生成样本...")
num_gen_samples = 16

# 1. 基线模型 + 无OT指导
print("1. 基线模型 + 无OT指导")
samples_baseline_no_ot = sample_with_ot_guidance(model_baseline, real_samples_for_guidance,
                                                 num_samples=num_gen_samples, use_ot_guidance=False)

# 2. 基线模型 + 有OT指导
print("2. 基线模型 + 有OT指导")
samples_baseline_with_ot = sample_with_ot_guidance(model_baseline, real_samples_for_guidance,
                                                   num_samples=num_gen_samples, use_ot_guidance=True)

# 3. OT模型 + 无OT指导
print("3. OT模型 + 无OT指导")
samples_ot_no_ot = sample_with_ot_guidance(model_with_ot, real_samples_for_guidance,
                                           num_samples=num_gen_samples, use_ot_guidance=False)

# 4. OT模型 + 有OT指导
print("4. OT模型 + 有OT指导")
samples_ot_with_ot = sample_with_ot_guidance(model_with_ot, real_samples_for_guidance,
                                             num_samples=num_gen_samples, use_ot_guidance=True)

# --- 9. 可视化所有结果 ---
plt.figure(figsize=(20, 12))

# 第一行：真实样本和训练损失
plt.subplot(3, 5, 1)
grid_real = make_grid(real_samples_for_guidance.cpu(), nrow=4, normalize=True, pad_value=1.0)
plt.imshow(grid_real.permute(1, 2, 0))
plt.axis('off')
plt.title(f'Real Samples (n={n_shot})')

# 训练损失曲线
plt.subplot(3, 5, 2)
if losses_ot is not None:
    # [优化] 使用移动平均使曲线更平滑
    def moving_average(a, n=100):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n


    plt.plot(moving_average(losses_ot), label='With OT Loss', alpha=0.7)
plt.plot(moving_average(losses_baseline), label='Baseline', alpha=0.7)
plt.xlabel('Iteration')
plt.ylabel('Loss (Smoothed)')
plt.legend()
plt.title('Training Loss')

# Sinkhorn损失曲线（如果存在）
plt.subplot(3, 5, 3)
if ot_losses_ot is not None:
    ot_losses_smoothed = moving_average([l for l in ot_losses_ot if l > 0])
    plt.plot(ot_losses_smoothed, label='OT Loss', color='orange', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('OT Loss (Smoothed)')
    plt.legend()
    plt.title('Sinkhorn Loss during Training')

# 第二行：基线模型的两种采样结果
plt.subplot(3, 5, 6)
grid_baseline_no_ot = make_grid(samples_baseline_no_ot[-1], nrow=4, normalize=True, pad_value=1.0)
plt.imshow(grid_baseline_no_ot.permute(1, 2, 0))
plt.axis('off')
plt.title('Baseline Model\nNo OT Guidance')

plt.subplot(3, 5, 7)
grid_baseline_with_ot = make_grid(samples_baseline_with_ot[-1], nrow=4, normalize=True, pad_value=1.0)
plt.imshow(grid_baseline_with_ot.permute(1, 2, 0))
plt.axis('off')
plt.title('Baseline Model\nWith OT Guidance')

# 第三行：OT模型的两种采样结果
plt.subplot(3, 5, 11)
grid_ot_no_ot = make_grid(samples_ot_no_ot[-1], nrow=4, normalize=True, pad_value=1.0)
plt.imshow(grid_ot_no_ot.permute(1, 2, 0))
plt.axis('off')
plt.title('OT-trained Model\nNo OT Guidance')

plt.subplot(3, 5, 12)
grid_ot_with_ot = make_grid(samples_ot_with_ot[-1], nrow=4, normalize=True, pad_value=1.0)
plt.imshow(grid_ot_with_ot.permute(1, 2, 0))
plt.axis('off')
plt.title('OT-trained Model\nWith OT Guidance')

# 添加说明文字
plt.figtext(0.1, 0.95, 'Training Phase', fontsize=14, fontweight='bold', ha='left')
plt.figtext(0.1, 0.63, 'Sampling Phase - Baseline Model', fontsize=14, fontweight='bold', ha='left')
plt.figtext(0.1, 0.30, 'Sampling Phase - OT-trained Model', fontsize=14, fontweight='bold', ha='left')

plt.tight_layout()
plt.savefig('all_results_comparison_optimized.png', dpi=150, bbox_inches='tight')
plt.show()

print("实验完成！所有结果已保存到 'all_results_comparison_optimized.png'")

# --- 10. 单独显示四种生成结果的大图 ---
plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.imshow(grid_baseline_no_ot.permute(1, 2, 0))
plt.axis('off')
plt.title('Baseline + No OT Guidance\n(Ordinary)')

plt.subplot(1, 4, 2)
plt.imshow(grid_baseline_with_ot.permute(1, 2, 0))
plt.axis('off')
plt.title('Baseline + OT Guidance\n(Use OT when sampling)')

plt.subplot(1, 4, 3)
plt.imshow(grid_ot_no_ot.permute(1, 2, 0))
plt.axis('off')
plt.title('OT-trained + No OT Guidance\n(Use OT when training)')
plt.subplot(1, 4, 4)

plt.subplot(1, 4, 4)
plt.imshow(grid_ot_with_ot.permute(1, 2, 0))
plt.axis('off')
plt.title('OT-trained + OT Guidance\n(Use OT in both)')

plt.tight_layout()
plt.savefig('generation_comparison_optimized.png', dpi=150, bbox_inches='tight')
plt.show()
# --- 11. 质量和多样性指标计算 ---
# 确保已安装 piq: pip install piq
try:
    import piq
    from itertools import combinations
    from torch.utils.data import Dataset  # [修复] 导入 Dataset
except ImportError:
    print("=" * 50)
    print("错误：未找到 'piq' 库。")
    print("请先运行: pip install piq")
    print("=" * 50)
    # 如果在Jupyter/Colab中，可以取消下一行的注释来自动安装
    # !pip install piq
    # import piq
    # from itertools import combinations
    # from torch.utils.data import Dataset
    raise ImportError("需要 piq 库来进行指标计算。")


# [修复] 新增一个 Dataset 类，以 piq 期望的字典格式返回数据
class FIDDataset(Dataset):
    def __init__(self, tensor_data):
        self.tensor_data = tensor_data

    def __len__(self):
        return len(self.tensor_data)

    def __getitem__(self, idx):
        # 返回 piq 期望的字典格式
        return {'images': self.tensor_data[idx]}


@torch.no_grad()
def unnormalize(images_tensor):
    """ 将图像从 [-1, 1] 转换回 [0, 1] """
    return (images_tensor + 1.0) / 2.0


@torch.no_grad()
def preprocess_for_fid_kid(images_tensor):
    """
    为 FID/KID 准备图像：
    1. Un-normalize to [0, 1]
    2. 扩展到 3 个通道
    3. Resize to (299, 299)
    """
    images = unnormalize(images_tensor)
    # 扩展为 3 通道
    if images.shape[1] == 1:
        images = images.expand(-1, 3, -1, -1)
    # Resize
    images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    # Clamp to [0, 1]
    return torch.clamp(images, 0.0, 1.0)


@torch.no_grad()
def preprocess_for_lpips(images_tensor, target_size=64):
    """
    为 LPIPS 准备图像：
    1. Un-normalize to [0, 1]
    2. 扩展到 3 个通道
    3. [修复] 将图像放大到 LPIPS 网络的最小尺寸 (e.g., 64x64)
    """
    images = unnormalize(images_tensor)
    # 扩展为 3 通道
    if images.shape[1] == 1:
        images = images.expand(-1, 3, -1, -1)

    # [修复] 添加上采样以避免 LPIPS 网络中的 0x0 错误
    images = F.interpolate(images, size=(target_size, target_size), mode='bilinear', align_corners=False)

    return torch.clamp(images, 0.0, 1.0)

# [修复] 替换此整个函数
@torch.no_grad()
def calculate_metrics(generated_sets, all_real_samples):
    """
    计算所有指标并返回一个字典。

    Args:
        generated_sets (dict): {'name': tensor, ...}
        all_real_samples (tensor): (N, 1, 28, 28)
    """
    # --- 1. 初始化指标 ---
    fid_metric = piq.FID().to(device)
    kid_metric = piq.KID().to(device)
    lpips_metric = piq.LPIPS(reduction='none', replace_pooling=True).to(device)

    # --- 2. 准备真实数据 ---
    print("正在为真实数据预计算 Inception 特征...")
    # 预处理真实图像以进行特征提取
    real_images_fid_kid = preprocess_for_fid_kid(all_real_samples)
    real_images_lpips = preprocess_for_lpips(all_real_samples)
    real_images_unnorm = unnormalize(all_real_samples)

    # 为 piq.compute_feats 创建 DataLoader
    real_fid_kid_dataset = FIDDataset(real_images_fid_kid)
    real_fid_kid_loader = DataLoader(real_fid_kid_dataset, batch_size=16)  # 使用一个批大小

    # 计算一次真实特征
    real_feats_fid = fid_metric.compute_feats(real_fid_kid_loader)
    real_feats_kid = kid_metric.compute_feats(real_fid_kid_loader)

    N_real = len(all_real_samples)
    results = {}

    # --- 3. 循环计算每个生成集 ---
    for name, gen_tensor in generated_sets.items():
        print(f"\n--- 正在计算 {name} 的指标 ---")
        N_gen = len(gen_tensor)

        # 准备生成的数据
        gen_images_fid_kid = preprocess_for_fid_kid(gen_tensor)
        gen_images_lpips = preprocess_for_lpips(gen_tensor)
        gen_images_unnorm = unnormalize(gen_tensor)

        # --- FID / KID ---
        print("计算 FID / KID...")

        # 为 piq.compute_feats 创建 DataLoader
        gen_fid_kid_dataset = FIDDataset(gen_images_fid_kid)
        gen_fid_kid_loader = DataLoader(gen_fid_kid_dataset, batch_size=16)

        gen_feats_fid = fid_metric.compute_feats(gen_fid_kid_loader)
        gen_feats_kid = kid_metric.compute_feats(gen_fid_kid_loader)

        fid_value = fid_metric(gen_feats_fid, real_feats_fid).item()
        kid_value = kid_metric(gen_feats_kid, real_feats_kid).item()

        # --- SSIM / PSNR (最近邻保真度) ---
        print("计算 SSIM / PSNR (最近邻)...")
        ssim_scores = []
        psnr_scores = []

        # 遍历每个生成的图像
        for i in tqdm(range(N_gen), desc="SSIM/PSNR"):
            g_img = gen_images_unnorm[i:i + 1]  # (1, 1, 28, 28)
            g_img_expanded = g_img.expand(N_real, -1, -1, -1)

            ssim_all_real = piq.ssim(g_img_expanded, real_images_unnorm, data_range=1.0, reduction='none')

            # [修复] 删除了 'downsample=False' 参数
            psnr_all_real = piq.psnr(g_img_expanded, real_images_unnorm, data_range=1.0, reduction='none')

            ssim_scores.append(ssim_all_real.max())
            psnr_scores.append(psnr_all_real.max())

        avg_ssim = torch.stack(ssim_scores).mean().item()
        avg_psnr = torch.stack(psnr_scores).mean().item()

        # --- Diversity (LPIPS) ---
        print("计算 Diversity (LPIPS)...")
        diversity_scores = []

        # 比较所有 (i, j) 对
        for i, j in tqdm(combinations(range(N_gen), 2), desc="Diversity", total=N_gen * (N_gen - 1) // 2):
            img_i = gen_images_lpips[i:i + 1]
            img_j = gen_images_lpips[j:j + 1]

            score = lpips_metric(img_i, img_j).item()
            diversity_scores.append(score)

        avg_diversity = np.mean(diversity_scores) if diversity_scores else 0.0

        # --- 存储结果 ---
        results[name] = {
            "FID": fid_value,
            "KID": kid_value,
            "SSIM": avg_ssim,
            "PSNR": avg_psnr,
            "Diversity (LPIPS)": avg_diversity
        }

    return results

# --- 主度量计算流程 ---
if __name__ == "__main__":
    N_METRICS = n_shot  # 生成 50 个样本进行评估
    print(f"\n--- 开始指标评估 (N={N_METRICS}) ---")

    # --- 1. 加载所有 50 个真实样本 ---
    all_real_samples = torch.stack(
        [few_shot_dataset[i][0] for i in range(len(few_shot_dataset))]
    ).to(device)

    # --- 2. 为指标重新生成 50 个样本 ---
    print("\n为指标评估重新生成样本...")

    print("1. 基线模型 + 无OT指导")
    gen_baseline_no_ot = sample_with_ot_guidance(model_baseline, all_real_samples,
                                                 num_samples=N_METRICS, use_ot_guidance=False)[-1].to(device)

    print("2. 基线模型 + 有OT指导")
    gen_baseline_with_ot = sample_with_ot_guidance(model_baseline, all_real_samples,
                                                   num_samples=N_METRICS, use_ot_guidance=True)[-1].to(device)

    print("3. OT模型 + 无OT指导")
    gen_ot_no_ot = sample_with_ot_guidance(model_with_ot, all_real_samples,
                                           num_samples=N_METRICS, use_ot_guidance=False)[-1].to(device)

    print("4. OT模型 + 有OT指导")
    gen_ot_with_ot = sample_with_ot_guidance(model_with_ot, all_real_samples,
                                             num_samples=N_METRICS, use_ot_guidance=True)[-1].to(device)

    generated_sets_for_metrics = {
        "Baseline (No Guide)": gen_baseline_no_ot,
        "Baseline (OT Guide)": gen_baseline_with_ot,
        "OT-Trained (No Guide)": gen_ot_no_ot,
        "OT-Trained (OT Guide)": gen_ot_with_ot
    }

    # --- 3. 计算所有指标 ---
    all_metrics = calculate_metrics(generated_sets_for_metrics, all_real_samples)

    # --- 4. 打印结果 ---
    print("\n\n" + "=" * 70)
    print(f"            少样本 (N={n_shot}) 扩散模型指标评估结果")
    print("=" * 70)

    # 打印表头
    header = f"{'Method':<24} | {'FID':<10} | {'KID (x1e3)':<12} | {'SSIM':<10} | {'PSNR':<10} | {'Diversity (LPIPS)':<18}"
    print(header)
    print("-" * len(header))
    print(" (FID/KID: 越低越好 | SSIM/PSNR/Diversity: 越高越好)")
    print("-" * len(header))

    # 打印每一行
    for name, metrics in all_metrics.items():
        print(f"{name:<24} | "
              f"{metrics['FID']:<10.3f} | "
              f"{metrics['KID'] * 1000:<12.4f} | "  # KID 通常很小，乘以1000
              f"{metrics['SSIM']:<10.3f} | "
              f"{metrics['PSNR']:<10.3f} | "
              f"{metrics['Diversity (LPIPS)']:<18.3f}")

    print("=" * 70)
import pandas as pd
import os
import datetime


# --- 12. 实验数据记录到 Excel/CSV ---

def log_experiment_to_csv(filepath, hyperparameters, metrics_results):
    """
    将本次实验的超参数和结果指标记录到一个 CSV 文件中。
    如果文件不存在，则创建它；如果存在，则追加新的一行。
    """

    # 1. 准备要记录的数据字典
    log_data = {}

    # 添加时间戳
    log_data['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 添加所有超参数
    log_data.update(hyperparameters)

    # 2. 扁平化指标字典
    # 将 {'Baseline (No Guide)': {'FID': 100, 'KID': 0.1}, ...}
    # 转换为 {'Baseline (No Guide) - FID': 100, 'Baseline (No Guide) - KID': 0.1, ...}
    for method_name, metrics in metrics_results.items():
        for metric_name, value in metrics.items():
            column_name = f"{method_name} - {metric_name}"
            log_data[column_name] = value

    # 3. 转换为 Pandas DataFrame
    new_log_df = pd.DataFrame([log_data])

    # 4. 读写文件
    if os.path.exists(filepath):
        print(f"\n正在追加记录到已有的文件: {filepath}")
        try:
            # 读取旧文件
            existing_df = pd.read_csv(filepath)
            # 合并新旧数据
            combined_df = pd.concat([existing_df, new_log_df], ignore_index=True)
        except pd.errors.EmptyDataError:
            print("CSV 文件为空，将创建新文件。")
            combined_df = new_log_df
        except Exception as e:
            print(f"读取 CSV 时出错: {e}。将覆盖为一个新文件。")
            combined_df = new_log_df
    else:
        print(f"\n未找到记录文件，将创建新文件: {filepath}")
        combined_df = new_log_df

    # 5. 保存回 CSV
    try:
        combined_df.to_csv(filepath, index=False)
        print("实验数据已成功记录！")
    except Exception as e:
        print(f"!!! 错误：无法保存到 CSV 文件。")
        print(f"错误详情: {e}")
        print("请检查文件权限或路径是否正确。")


# --- 主日志记录流程 ---
if __name__ == "__main__":

    # --- 1. 定义要记录的超参数 ---
    # (这些变量必须在你的主脚本中已经定义)
    hyperparams_to_log = {
        "target_class": target_class,
        "n_shot": n_shot,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "T_steps": T,
        "unet_time_dim": 32,  # 这是我们新U-Net中硬编码的
        "ot_loss_weight": ot_loss_weight,
        "ot_guidance_weight": ot_guidance_weight,
        "sinkhorn_epsilon": sinkhorn_epsilon,
        "sinkhorn_n_iter": sinkhorn_n_iter
    }

    # --- 2. 定义日志文件路径 ---
    log_file_path = "experiment_log_mnist.csv"

    # --- 3. 调用日志函数 ---
    # (假设 all_metrics 变量已经从之前的指标计算中获取)
    if 'all_metrics' in locals():
        log_experiment_to_csv(log_file_path, hyperparams_to_log, all_metrics)
    else:
        print("\n[日志记录跳过]：未找到 'all_metrics' 变量。请确保指标计算已运行。")