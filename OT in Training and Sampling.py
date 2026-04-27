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
batch_size = 32
num_epochs = 500
T = 200
beta_start = 0.0001
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T, device=device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# Sinkhorn 参数
sinkhorn_epsilon = 0.01
sinkhorn_n_iter = 5
ot_loss_weight = 0.1
ot_guidance_weight = 2.0  # 采样时OT指导的权重

# 少样本参数
target_class = 5
n_shot = 500
total_samples= 50000
# --- 2. 加载并准备少样本MNIST数据 ---
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

idx = (train_dataset.targets == target_class).nonzero(as_tuple=True)[0]
selected_idx = idx[torch.randperm(len(idx))[:n_shot]]
few_shot_dataset = Subset(train_dataset, selected_idx)
train_loader = DataLoader(few_shot_dataset, batch_size=batch_size, shuffle=True)

# 获取一些真实样本用于OT指导
real_samples_for_guidance = next(iter(DataLoader(few_shot_dataset, batch_size=16, shuffle=True)))[0].to(device)


# --- 3. 定义神经网络 ---
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.initial_conv = nn.Conv2d(2, 32, 3, stride=1, padding=1)

        self.encoder = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.mid = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, stride=1, padding=1),
        )

    def forward(self, x, t):
        t_emb = t.float() / T
        t_emb = t_emb.view(-1, 1, 1, 1).expand(x.size(0), 1, x.size(2), x.size(3))
        x_with_time = torch.cat([x, t_emb], dim=1)

        x0 = self.initial_conv(x_with_time)
        x1 = self.encoder(x0)
        x2 = self.mid(x1)
        out = self.decoder(x2)
        return out


# --- 4. 定义Sinkhorn算法 ---
def sinkhorn_loss(x_real, x_fake, epsilon=sinkhorn_epsilon, n_iters=sinkhorn_n_iter):
    n = x_real.size(0)
    m = x_fake.size(0)

    x_real_flat = x_real.view(n, -1)
    x_fake_flat = x_fake.view(m, -1)

    C = torch.cdist(x_real_flat, x_fake_flat, p=2) ** 2
    C = C / (torch.max(C) + 1e-8)

    K = torch.exp(-C / epsilon)
    u = torch.ones(n, device=device) / n
    v = torch.ones(m, device=device) / m

    for _ in range(n_iters):
        u = 1.0 / (torch.mm(K, v.unsqueeze(1)).squeeze() + 1e-8)
        v = 1.0 / (torch.mm(K.t(), u.unsqueeze(1)).squeeze() + 1e-8)

    P = torch.diag(u) @ K @ torch.diag(v)
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
        if use_ot_guidance and t < T * 0.7:  # 只在前期使用OT指导，避免后期过度干扰
            # 需要计算关于x的梯度，所以启用梯度计算
            x_requires_grad = x.detach().requires_grad_(True)

            # 预测当前时间步的x0
            with torch.enable_grad():
                predicted_noise_grad = model(x_requires_grad, t_batch)
                predicted_x0 = (x_requires_grad - sqrt_one_minus_alphas_cumprod[t] * predicted_noise_grad) / \
                               sqrt_alphas_cumprod[t]
                predicted_x0 = torch.clamp(predicted_x0, -1.0, 1.0)

                # 计算OT损失
                ot_loss = sinkhorn_loss(real_samples[:num_samples], predicted_x0)

                # 计算OT损失关于x的梯度
                ot_grad = torch.autograd.grad(ot_loss, x_requires_grad)[0]

            # 使用OT梯度调整均值
            mean = mean - ot_guidance_weight * ot_grad

        if t > 0:
            # 计算方差并添加噪声
            variance = betas[t]
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(variance) * noise
        else:
            # 最后一步不需要添加噪声
            x = mean

        x = torch.clamp(x, -1.0, 1.0)

        if t % 40 == 0 or t == 0:
            samples.append(x.cpu())

    return samples


# --- 6. 训练函数 ---
def train_model(use_ot_loss=False, model_name="model"):
    if use_ot_loss:
        model = SimpleUNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        ot_losses = []
        mse_losses = []

        print(f"训练{model_name}（使用Sinkhorn损失）...")
        for epoch in range(num_epochs):
            model.train()
            pbar = tqdm(train_loader, desc=f'{model_name} Epoch {epoch + 1}/{num_epochs}')
            for i, (images, _) in enumerate(pbar):
                optimizer.zero_grad()
                images = images.to(device)
                batch_size = images.size(0)

                t = torch.randint(0, T, (batch_size,), device=device).long()
                noise = torch.randn_like(images)
                sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
                sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
                noisy_images = sqrt_alpha_cumprod_t * images + sqrt_one_minus_alpha_cumprod_t * noise

                predicted_noise = model(noisy_images, t)
                mse_loss = F.mse_loss(predicted_noise, noise)

                if epoch > 5:
                    with torch.no_grad():
                        predicted_x0 = (noisy_images - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / sqrt_alpha_cumprod_t
                        predicted_x0 = torch.clamp(predicted_x0, -1.0, 1.0)

                    if batch_size > 8:
                        real_subset = images[:8]
                        fake_subset = predicted_x0[:8]
                        ot_loss = sinkhorn_loss(real_subset, fake_subset)
                    else:
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
                    'OT Loss': f'{ot_loss.item():.4f}' if ot_loss != 0 else '0.0000'
                })

        return model, losses, mse_losses, ot_losses

    else:
        # 训练基线模型（无OT损失）
        model = SimpleUNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        losses = []

        print(f"训练{model_name}（基线，无Sinkhorn损失）...")
        for epoch in range(num_epochs):
            model.train()
            for images, _ in train_loader:
                images = images.to(device)
                batch_size = images.size(0)

                t = torch.randint(0, T, (batch_size,), device=device).long()
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

        return model, losses, None, None


# --- 7. 主实验 ---
# 训练两个模型：一个使用OT损失，一个不使用
model_with_ot, losses_ot, mse_losses_ot, ot_losses_ot = train_model(use_ot_loss=True, model_name="OT模型")
model_baseline, losses_baseline, _, _ = train_model(use_ot_loss=False, model_name="基线模型")

# --- 8. 生成四种情况的样本 ---
print("\n生成样本...")

# 1. 基线模型 + 无OT指导
print("1. 基线模型 + 无OT指导")
samples_baseline_no_ot = sample_with_ot_guidance(model_baseline, real_samples_for_guidance,
                                                 num_samples=16, use_ot_guidance=False)

# 2. 基线模型 + 有OT指导
print("2. 基线模型 + 有OT指导")
samples_baseline_with_ot = sample_with_ot_guidance(model_baseline, real_samples_for_guidance,
                                                   num_samples=16, use_ot_guidance=True)

# 3. OT模型 + 无OT指导
print("3. OT模型 + 无OT指导")
samples_ot_no_ot = sample_with_ot_guidance(model_with_ot, real_samples_for_guidance,
                                           num_samples=16, use_ot_guidance=False)

# 4. OT模型 + 有OT指导
print("4. OT模型 + 有OT指导")
samples_ot_with_ot = sample_with_ot_guidance(model_with_ot, real_samples_for_guidance,
                                             num_samples=16, use_ot_guidance=True)

# --- 9. 可视化所有结果 ---
plt.figure(figsize=(20, 12))

# 第一行：真实样本和训练损失
plt.subplot(3, 5, 1)
grid_real = make_grid(real_samples_for_guidance.cpu(), nrow=4, normalize=True, pad_value=1.0)
plt.imshow(grid_real.permute(1, 2, 0))
plt.axis('off')
plt.title('Real Samples\n(n=50)')

# 训练损失曲线
plt.subplot(3, 5, 2)
if losses_ot is not None:
    plt.plot(losses_ot, label='With OT Loss', alpha=0.7)
plt.plot(losses_baseline, label='Baseline', alpha=0.7)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss')

# Sinkhorn损失曲线（如果存在）
plt.subplot(3, 5, 3)
if ot_losses_ot is not None:
    plt.plot(ot_losses_ot, label='OT Loss', color='orange', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('OT Loss')
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
plt.savefig('all_results_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("实验完成！所有结果已保存到 'all_results_comparison.png'")

# --- 10. 单独显示四种生成结果的大图 ---
plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.imshow(grid_baseline_no_ot.permute(1, 2, 0))
plt.axis('off')
plt.title('Baseline + No OT Guidance\n Ordinary')

plt.subplot(1, 4, 2)
plt.imshow(grid_baseline_with_ot.permute(1, 2, 0))
plt.axis('off')
plt.title('Baseline + OT Guidance\nUse OT when sampling')

plt.subplot(1, 4, 3)
plt.imshow(grid_ot_no_ot.permute(1, 2, 0))
plt.axis('off')
plt.title('OT-trained + No OT Guidance\nUse OT when training')
plt.subplot(1, 4, 4)

plt.subplot(1, 4, 4)
plt.imshow(grid_ot_with_ot.permute(1, 2, 0))
plt.axis('off')
plt.title('OT-trained + OT Guidance\nUse OT both when sampling and training')

plt.tight_layout()
plt.savefig('generation_comparison.png', dpi=150, bbox_inches='tight')
plt.show()