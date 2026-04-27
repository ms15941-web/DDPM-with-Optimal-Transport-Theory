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
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
batch_size = 32
num_epochs = 50
T = 1000
beta_start = 0.0001
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T, device=device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)  # 修复：添加前一个alpha累积
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# Sinkhorn 参数
sinkhorn_epsilon = 0.01
sinkhorn_n_iter = 5
ot_loss_weight = 0.1

# 少样本参数
target_class = 1
n_shot = 50000

# --- 2. 加载并准备少样本MNIST数据 ---
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

idx = (train_dataset.targets == target_class).nonzero(as_tuple=True)[0]
selected_idx = idx[torch.randperm(len(idx))[:n_shot]]
few_shot_dataset = Subset(train_dataset, selected_idx)
train_loader = DataLoader(few_shot_dataset, batch_size=batch_size, shuffle=True)


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


model = SimpleUNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# --- 4. 定义Sinkhorn算法 ---
def sinkhorn_loss(x_real, x_fake, epsilon=sinkhorn_epsilon, n_iters=sinkhorn_n_iter):
    n = x_real.size(0)
    m = x_fake.size(0)

    x_real_flat = x_real.view(n, -1)
    x_fake_flat = x_fake.view(m, -1)

    C = torch.cdist(x_real_flat, x_fake_flat, p=2) ** 2
    C = C / torch.max(C)

    K = torch.exp(-C / epsilon)
    u = torch.ones(n, device=device) / n
    v = torch.ones(m, device=device) / m

    for _ in range(n_iters):
        u = 1.0 / (torch.mm(K, v.unsqueeze(1)).squeeze() + 1e-8)
        v = 1.0 / (torch.mm(K.t(), u.unsqueeze(1)).squeeze() + 1e-8)

    P = torch.diag(u) @ K @ torch.diag(v)
    ot_distance = torch.sum(P * C)

    return ot_distance


# --- 5. 训练循环 ---
losses = []
ot_losses = []
mse_losses = []

print("开始训练...")
for epoch in range(num_epochs):
    model.train()
    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
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


# --- 6. 修复采样过程 ---
@torch.no_grad()
def sample(model, num_samples=16):
    model.eval()
    x = torch.randn(num_samples, 1, 28, 28, device=device)
    samples = []

    # 修复：使用正确的DDPM采样算法
    for t in tqdm(range(T - 1, -1, -1), desc="Sampling"):
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
        predicted_noise = model(x, t_batch)

        # DDPM采样公式
        alpha_t = alphas[t]
        alpha_cumprod_t = alphas_cumprod[t]
        beta_t = betas[t]

        if t > 0:
            # 计算均值
            mean = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise)
            # 计算方差
            variance = betas[t]
            # 添加噪声
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(variance) * noise
        else:
            # 最后一步不需要添加噪声
            x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise)

        x = torch.clamp(x, -1.0, 1.0)

        if t % 40 == 0 or t == 0:  # 减少存储频率
            samples.append(x.cpu())

    return samples


print("生成样本...")
generated_samples = sample(model, num_samples=16)

# --- 7. 可视化结果 ---
if True:
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(losses, label='Total Loss', alpha=0.7)
    plt.plot(mse_losses, label='MSE Loss', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')

    plt.subplot(1, 3, 2)
    plt.plot(ot_losses, label='OT Loss', color='orange', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('OT Loss')
    plt.legend()
    plt.title('Sinkhorn Loss')

    plt.subplot(1, 3, 3)
    grid = make_grid(generated_samples[-1], nrow=4, normalize=True, pad_value=1.0)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title('Generated Digits')

    plt.tight_layout()
    plt.savefig('results.png')
    plt.show()

    # 显示真实训练样本
    real_samples = next(iter(DataLoader(few_shot_dataset, batch_size=16, shuffle=True)))[0]
    grid_real = make_grid(real_samples, nrow=4, normalize=True, pad_value=1.0)
    plt.figure(figsize=(6, 6))
    plt.imshow(grid_real.permute(1, 2, 0))
    plt.axis('off')
    plt.title(f'Real Training Samples (n={n_shot})')
    plt.savefig('real_samples.png')
    plt.show()

    print("实验完成！")

# 对比实验：无Sinkhorn损失的基线模型
print("\n训练基线模型（无Sinkhorn损失）...")
baseline_model = SimpleUNet().to(device)
baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=1e-3)

for epoch in range(num_epochs):  # 训练较少轮数
    baseline_model.train()
    for images, _ in train_loader:
        images = images.to(device)
        batch_size = images.size(0)

        t = torch.randint(0, T, (batch_size,), device=device).long()
        noise = torch.randn_like(images)
        sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        noisy_images = sqrt_alpha_cumprod_t * images + sqrt_one_minus_alpha_cumprod_t * noise

        predicted_noise = baseline_model(noisy_images, t)
        loss = F.mse_loss(predicted_noise, noise)

        baseline_optimizer.zero_grad()
        loss.backward()
        baseline_optimizer.step()

print("生成基线模型样本...")
baseline_samples = sample(baseline_model, num_samples=16)

# 对比显示结果
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
grid_real = make_grid(real_samples, nrow=4, normalize=True, pad_value=1.0)
plt.imshow(grid_real.permute(1, 2, 0))
plt.axis('off')
plt.title('Real Samples')

plt.subplot(1, 3, 2)
grid_ot = make_grid(generated_samples[-1], nrow=4, normalize=True, pad_value=1.0)
plt.imshow(grid_ot.permute(1, 2, 0))
plt.axis('off')
plt.title('With Sinkhorn Loss')

plt.subplot(1, 3, 3)
grid_baseline = make_grid(baseline_samples[-1], nrow=4, normalize=True, pad_value=1.0)
plt.imshow(grid_baseline.permute(1, 2, 0))
plt.axis('off')
plt.title('Baseline (No Sinkhorn)')

plt.tight_layout()
plt.savefig('comparison.png')
plt.show()

print("对比实验完成！")


# --- 8. 图像质量评估函数 (修复单通道问题) ---
def evaluate_generated_images(real_images, generated_images, model_name="Model"):
    """
    评估生成图像的质量 - 专门针对单通道图像优化
    real_images: 真实图像张量 (N, 1, H, W)
    generated_images: 生成图像张量 (N, 1, H, W)
    model_name: 模型名称
    """
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    import torch

    print(f"\n=== {model_name} 评估结果 ===")

    # 确保图像在[0,1]范围内
    real_images = (real_images + 1) / 2  # 从[-1,1]转换到[0,1]
    generated_images = (generated_images + 1) / 2

    # 限制样本数量以避免内存问题
    max_samples = min(64, len(real_images), len(generated_images))
    real_images = real_images[:max_samples]
    generated_images = generated_images[:max_samples]

    # 1. FID分数 (单通道转三通道)
    try:
        # 将单通道图像复制为三通道
        real_rgb = real_images.repeat(1, 3, 1, 1)
        generated_rgb = generated_images.repeat(1, 3, 1, 1)

        # 调整图像大小为Inception网络需要的尺寸(299x299)
        resize_transform = transforms.Resize((299, 299), antialias=True)
        real_resized = resize_transform(real_rgb)
        generated_resized = resize_transform(generated_rgb)

        # 将图像转换为uint8格式
        real_uint8 = (real_resized * 255).byte()
        generated_uint8 = (generated_resized * 255).byte()

        fid = FrechetInceptionDistance(feature=64, normalize=False)
        fid.update(real_uint8, real=True)
        fid.update(generated_uint8, real=False)
        fid_score = fid.compute().item()
        print(f"FID分数: {fid_score:.4f} (越低越好)")
    except Exception as e:
        fid_score = float('inf')
        print(f"FID计算失败: {e}")

    # 2. KID分数 (单通道转三通道)
    try:
        # 使用上面已经转换的图像
        kid = KernelInceptionDistance(subset_size=min(50, max_samples), normalize=False)
        kid.update(real_uint8, real=True)
        kid.update(generated_uint8, real=False)
        kid_mean, kid_std = kid.compute()
        print(f"KID分数: {kid_mean.item():.4f} ± {kid_std.item():.4f} (越低越好)")
    except Exception as e:
        kid_mean = float('inf')
        print(f"KID计算失败: {e}")

    # 3. 图像清晰度评估 (PSNR - 越高越好) - 使用原始单通道图像
    try:
        psnr = PeakSignalNoiseRatio()
        psnr_scores = []
        # 随机选择样本对计算PSNR
        for i in range(min(20, len(real_images))):
            real_img = real_images[i].unsqueeze(0)
            gen_img = generated_images[i].unsqueeze(0)
            psnr_score = psnr(gen_img, real_img)
            psnr_scores.append(psnr_score.item())
        avg_psnr = np.mean(psnr_scores)
        std_psnr = np.std(psnr_scores)
        print(f"PSNR: {avg_psnr:.4f} ± {std_psnr:.4f} dB (越高越好)")
    except Exception as e:
        avg_psnr = 0
        std_psnr = 0
        print(f"PSNR计算失败: {e}")

    # 4. 结构相似性 (SSIM - 越高越好) - 使用原始单通道图像
    try:
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        ssim_scores = []
        for i in range(min(20, len(real_images))):
            real_img = real_images[i].unsqueeze(0)
            gen_img = generated_images[i].unsqueeze(0)
            ssim_score = ssim(gen_img, real_img)
            ssim_scores.append(ssim_score.item())
        avg_ssim = np.mean(ssim_scores)
        std_ssim = np.std(ssim_scores)
        print(f"SSIM: {avg_ssim:.4f} ± {std_ssim:.4f} (越高越好)")
    except Exception as e:
        avg_ssim = 0
        std_ssim = 0
        print(f"SSIM计算失败: {e}")

    # 5. 多样性评估 (计算生成图像之间的平均距离)
    try:
        # 将生成图像展平
        gen_flat = generated_images.view(generated_images.size(0), -1)
        # 计算所有生成图像对之间的L2距离
        distances = torch.cdist(gen_flat, gen_flat, p=2)
        # 取上三角部分（不包括对角线）
        mask = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
        valid_distances = distances[mask]
        diversity = torch.mean(valid_distances) if len(valid_distances) > 0 else torch.tensor(0.0)
        diversity_std = torch.std(valid_distances) if len(valid_distances) > 0 else torch.tensor(0.0)
        print(f"多样性: {diversity.item():.4f} ± {diversity_std.item():.4f} (越高越好)")
    except Exception as e:
        diversity = 0
        diversity_std = 0
        print(f"多样性计算失败: {e}")

    # 6. 模式崩溃检测 (计算最近邻距离比率)
    try:
        real_flat = real_images.view(real_images.size(0), -1)
        gen_flat = generated_images.view(generated_images.size(0), -1)

        # 计算每个生成样本到最近真实样本的距离
        dist_to_real = torch.cdist(gen_flat, real_flat, p=2)
        min_dist_to_real = torch.min(dist_to_real, dim=1)[0]

        # 计算每个生成样本到最近其他生成样本的距离
        dist_to_gen = torch.cdist(gen_flat, gen_flat, p=2)
        # 创建掩码排除对角线
        mask = torch.eye(dist_to_gen.size(0), device=dist_to_gen.device, dtype=torch.bool)
        dist_to_gen_masked = dist_to_gen.clone()
        dist_to_gen_masked[mask] = float('inf')
        min_dist_to_gen = torch.min(dist_to_gen_masked, dim=1)[0]

        # 计算比率 (避免除零)
        safe_min_dist_to_gen = torch.where(min_dist_to_gen == 0, torch.tensor(1e-8, device=device), min_dist_to_gen)
        collapse_ratio = torch.mean(min_dist_to_real / safe_min_dist_to_gen)
        collapse_std = torch.std(min_dist_to_real / safe_min_dist_to_gen)
        print(f"模式崩溃: {collapse_ratio.item():.4f} ± {collapse_std.item():.4f} (接近1较好)")
    except Exception as e:
        collapse_ratio = 0
        collapse_std = 0
        print(f"模式崩溃检测失败: {e}")

    # 7. 图像锐度评估 (使用拉普拉斯方差)
    try:
        import cv2
        sharpness_scores = []
        for i in range(len(generated_images)):
            # 转换为numpy数组
            img = generated_images[i].squeeze().cpu().numpy()
            # 计算拉普拉斯方差
            laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
            sharpness_scores.append(laplacian_var)
        avg_sharpness = np.mean(sharpness_scores)
        std_sharpness = np.std(sharpness_scores)
        print(f"锐度: {avg_sharpness:.4f} ± {std_sharpness:.4f} (越高越好)")
    except Exception as e:
        avg_sharpness = 0
        std_sharpness = 0
        print(f"锐度计算失败: {e}")

    return {
        'fid': fid_score,
        'kid': kid_mean.item() if isinstance(kid_mean, torch.Tensor) else kid_mean,
        'psnr': avg_psnr,
        'psnr_std': std_psnr,
        'ssim': avg_ssim,
        'ssim_std': std_ssim,
        'diversity': diversity.item() if isinstance(diversity, torch.Tensor) else diversity,
        'diversity_std': diversity_std.item() if isinstance(diversity_std, torch.Tensor) else diversity_std,
        'collapse_ratio': collapse_ratio.item() if isinstance(collapse_ratio, torch.Tensor) else collapse_ratio,
        'collapse_std': collapse_std.item() if isinstance(collapse_std, torch.Tensor) else collapse_std,
        'sharpness': avg_sharpness,
        'sharpness_std': std_sharpness,
        'model_name': model_name
    }


def compare_models_evaluation(real_images, model_results):
    """
    比较多个模型的评估结果
    """
    evaluations = []

    for generated_images, model_name in model_results:
        eval_result = evaluate_generated_images(real_images.clone(), generated_images.clone(), model_name)
        evaluations.append(eval_result)

    # 打印比较表格
    print("\n" + "=" * 100)
    print("模型性能比较总结")
    print("=" * 100)
    print(
        f"{'模型名称':<15} {'FID':<8} {'KID':<10} {'PSNR':<10} {'SSIM':<8} {'多样性':<8} {'模式崩溃':<10} {'锐度':<8}")
    print("-" * 100)

    for eval_result in evaluations:
        print(f"{eval_result['model_name']:<15} "
              f"{eval_result['fid']:<8.2f} "
              f"{eval_result['kid']:<10.4f} "
              f"{eval_result['psnr']:<10.4f} "
              f"{eval_result['ssim']:<8.4f} "
              f"{eval_result['diversity']:<8.4f} "
              f"{eval_result['collapse_ratio']:<10.4f} "
              f"{eval_result['sharpness']:<8.4f}")

    # 可视化比较结果
    metrics = ['fid', 'kid', 'psnr', 'ssim', 'diversity', 'sharpness']
    metric_names = ['FID (越低越好)', 'KID (越低越好)', 'PSNR (越高越好)', 'SSIM (越高越好)', '多样性 (越高越好)',
                    '锐度 (越高越好)']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        if i < len(axes):
            values = [e[metric] for e in evaluations]
            names = [e['model_name'] for e in evaluations]

            # 对于FID和KID，数值越小越好，我们取倒数来可视化
            if metric in ['fid', 'kid']:
                values = [1 / (v + 1e-8) for v in values]  # 避免除零
                name = f"1/{name}"

            bars = axes[i].bar(names, values, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.8)
            axes[i].set_title(f'{name}', fontsize=12, fontweight='bold')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(axis='y', alpha=0.3)

            # 在柱状图上显示数值
            for bar, v in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width() / 2., height + 0.01 * max(values),
                             f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    return evaluations


# --- 9. 使用评估函数比较模型 ---
print("\n开始评估模型性能...")

# 准备真实样本用于评估
real_images_for_eval = next(iter(DataLoader(few_shot_dataset, batch_size=64, shuffle=True)))[0]

# 准备模型生成结果
model_results = [
    (generated_samples[-1], "Sinkhorn模型"),
    (baseline_samples[-1], "基线模型")
]

# 执行评估比较
evaluation_results = compare_models_evaluation(real_images_for_eval, model_results)

print("\n评估完成！")


# --- 10. 生成详细分析报告 ---
def generate_analysis_report(evaluation_results, real_images, generated_images_dict):
    """
    生成详细的分析报告
    """
    print("\n" + "=" * 80)
    print("详细分析报告")
    print("=" * 80)

    best_model = min(evaluation_results, key=lambda x: x['fid'])
    print(f"最佳模型: {best_model['model_name']} (基于FID分数)")

    # 分析每个指标的意义
    print("\n指标解释:")
    print("• FID: 衡量生成图像与真实图像的分布距离，越低表示质量越好")
    print("• KID: FID的无偏估计，越低越好")
    print("• PSNR: 峰值信噪比，衡量图像重建质量，越高越好")
    print("• SSIM: 结构相似性，衡量视觉质量，越高越好")
    print("• 多样性: 生成图像之间的差异程度，越高表示多样性越好")
    print("• 模式崩溃: 接近1表示正常，过大可能陷入模式崩溃")
    print("• 锐度: 图像清晰度，越高表示图像越清晰")

    # 可视化生成样本对比
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 真实样本
    real_grid = make_grid(real_images[:16], nrow=4, normalize=True, pad_value=1.0)
    axes[0, 0].imshow(real_grid.permute(1, 2, 0), cmap='gray')
    axes[0, 0].set_title('真实样本', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # 各模型生成样本
    for idx, (model_name, images) in enumerate(generated_images_dict.items()):
        row, col = (idx + 1) // 2, (idx + 1) % 2
        gen_grid = make_grid(images[:16], nrow=4, normalize=True, pad_value=1.0)
        axes[row, col].imshow(gen_grid.permute(1, 2, 0), cmap='gray')
        axes[row, col].set_title(f'{model_name}', fontsize=14, fontweight='bold')
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig('detailed_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


# 生成详细报告
generated_images_dict = {
    "Sinkhorn模型": generated_samples[-1],
    "基线模型": baseline_samples[-1]
}
generate_analysis_report(evaluation_results, real_images_for_eval, generated_images_dict)
if True:
    # --- 11. 数据记录功能 ---
    import pandas as pd
    import os
    from datetime import datetime
    import json


    def save_experiment_to_csv(hyperparameters, evaluation_results, csv_path='datacollection.csv'):
        """
        将实验超参数和评估结果保存到CSV文件

        Args:
            hyperparameters: 字典，包含超参数
            evaluation_results: 列表，包含每个模型的评估结果
            csv_path: CSV文件路径
        """
        # 创建数据目录（如果不存在）
        os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)

        # 准备数据行
        rows = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for eval_result in evaluation_results:
            row = {
                'timestamp': timestamp,
                'model_name': eval_result['model_name'],
                # 超参数
                'batch_size': hyperparameters.get('batch_size', ''),
                'num_epochs': hyperparameters.get('num_epochs', ''),
                'T': hyperparameters.get('T', ''),
                'learning_rate': hyperparameters.get('learning_rate', ''),
                'ot_loss_weight': hyperparameters.get('ot_loss_weight', ''),
                'n_shot': hyperparameters.get('n_shot', ''),
                'target_class': hyperparameters.get('target_class', ''),
                # 评估指标
                'fid': eval_result.get('fid', ''),
                'kid': eval_result.get('kid', ''),
                'psnr': eval_result.get('psnr', ''),
                'psnr_std': eval_result.get('psnr_std', ''),
                'ssim': eval_result.get('ssim', ''),
                'ssim_std': eval_result.get('ssim_std', ''),
                'diversity': eval_result.get('diversity', ''),
                'diversity_std': eval_result.get('diversity_std', ''),
                'collapse_ratio': eval_result.get('collapse_ratio', ''),
                'collapse_std': eval_result.get('collapse_std', ''),
                'sharpness': eval_result.get('sharpness', ''),
                'sharpness_std': eval_result.get('sharpness_std', ''),
                'total_training_time': hyperparameters.get('total_training_time', ''),
                'device': hyperparameters.get('device', '')
            }
            rows.append(row)

        # 创建DataFrame
        df_new = pd.DataFrame(rows)

        # 如果文件已存在，读取并追加新数据
        if os.path.exists(csv_path):
            df_existing = pd.read_csv(csv_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new

        # 保存到CSV
        df_combined.to_csv(csv_path, index=False)
        print(f"\n实验数据已保存到: {csv_path}")
        print(f"本次添加了 {len(rows)} 条记录")

        return df_combined


    def print_experiment_summary(hyperparameters, evaluation_results):
        """
        打印实验摘要
        """
        print("\n" + "=" * 80)
        print("实验摘要")
        print("=" * 80)
        print("超参数:")
        for key, value in hyperparameters.items():
            if key != 'total_training_time':  # 训练时间单独显示
                print(f"  {key}: {value}")

        print(f"\n训练时间: {hyperparameters.get('total_training_time', 'N/A')}")
        print(f"运行设备: {hyperparameters.get('device', 'N/A')}")

        print("\n最佳模型结果:")
        best_model = min(evaluation_results, key=lambda x: x['fid'] if x['fid'] != float('inf') else float('inf'))
        print(f"  模型: {best_model['model_name']}")
        print(f"  FID: {best_model['fid']:.4f}")
        print(f"  PSNR: {best_model['psnr']:.4f}")
        print(f"  多样性: {best_model['diversity']:.4f}")


    # --- 12. 修改主程序以记录数据 ---
    import time

    # 在训练开始前记录开始时间
    start_time = time.time()

    # ... 原有的所有代码保持不变 ...

    # 在评估完成后，添加数据记录部分
    print("\n开始记录实验数据...")

    # 收集超参数
    hyperparameters = {
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'T': T,
        'learning_rate': 1e-3,
        'ot_loss_weight': ot_loss_weight,
        'n_shot': n_shot,
        'target_class': target_class,
        'total_training_time': f"{time.time() - start_time:.2f}秒",
        'device': str(device)
    }

    # 打印实验摘要
    print_experiment_summary(hyperparameters, evaluation_results)

    # 保存到CSV文件
    csv_path = 'datacollection.csv'
    df = save_experiment_to_csv(hyperparameters, evaluation_results, csv_path)


    # --- 13. 添加数据分析功能 ---
    def analyze_experiment_history(csv_path='datacollection.csv'):
        """
        分析实验历史数据
        """
        if not os.path.exists(csv_path):
            print(f"数据文件 {csv_path} 不存在")
            return

        df = pd.read_csv(csv_path)
        print(f"\n{'=' * 80}")
        print("实验历史分析")
        print(f"{'=' * 80}")
        print(f"总实验次数: {len(df['timestamp'].unique())}")
        print(f"总记录数: {len(df)}")
        print(f"包含的模型: {df['model_name'].unique().tolist()}")

        # 按模型分组统计
        model_stats = df.groupby('model_name').agg({
            'fid': ['mean', 'min', 'std'],
            'psnr': ['mean', 'max', 'std'],
            'diversity': ['mean', 'max', 'std'],
            'ssim': ['mean', 'max', 'std']
        }).round(4)

        print(f"\n各模型平均性能:")
        print(model_stats)

        # 找到最佳实验配置
        if 'fid' in df.columns:
            best_fid_idx = df['fid'].idxmin()
            best_experiment = df.loc[best_fid_idx]
            print(f"\n最佳实验配置 (最低FID):")
            print(f"  时间: {best_experiment['timestamp']}")
            print(f"  模型: {best_experiment['model_name']}")
            print(f"  FID: {best_experiment['fid']:.4f}")
            print(f"  超参数: batch_size={best_experiment['batch_size']}, "
                  f"lr={best_experiment['learning_rate']}, "
                  f"ot_weight={best_experiment['ot_loss_weight']}")

        return df


    # 分析历史数据
    try:
        history_df = analyze_experiment_history(csv_path)

        # 可视化历史趋势（如果有足够数据）
        if len(history_df) > 1:
            plt.figure(figsize=(15, 10))

            # FID趋势
            plt.subplot(2, 3, 1)
            for model in history_df['model_name'].unique():
                model_data = history_df[history_df['model_name'] == model]
                plt.plot(model_data['timestamp'], model_data['fid'], 'o-', label=model, markersize=4)
            plt.title('FID趋势 (越低越好)')
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)

            # PSNR趋势
            plt.subplot(2, 3, 2)
            for model in history_df['model_name'].unique():
                model_data = history_df[history_df['model_name'] == model]
                plt.plot(model_data['timestamp'], model_data['psnr'], 'o-', label=model, markersize=4)
            plt.title('PSNR趋势 (越高越好)')
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 多样性趋势
            plt.subplot(2, 3, 3)
            for model in history_df['model_name'].unique():
                model_data = history_df[history_df['model_name'] == model]
                plt.plot(model_data['timestamp'], model_data['diversity'], 'o-', label=model, markersize=4)
            plt.title('多样性趋势 (越高越好)')
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 超参数影响分析
            if 'ot_loss_weight' in history_df.columns:
                plt.subplot(2, 3, 4)
                scatter = plt.scatter(history_df['ot_loss_weight'], history_df['fid'],
                                      c=history_df.index, cmap='viridis', alpha=0.7)
                plt.colorbar(scatter, label='实验序号')
                plt.xlabel('OT Loss Weight')
                plt.ylabel('FID')
                plt.title('OT权重 vs FID')
                plt.grid(True, alpha=0.3)

            # 批次大小影响
            if 'batch_size' in history_df.columns:
                plt.subplot(2, 3, 5)
                scatter = plt.scatter(history_df['batch_size'], history_df['psnr'],
                                      c=history_df.index, cmap='plasma', alpha=0.7)
                plt.colorbar(scatter, label='实验序号')
                plt.xlabel('Batch Size')
                plt.ylabel('PSNR')
                plt.title('批次大小 vs PSNR')
                plt.grid(True, alpha=0.3)

            plt.tight_layout()
            #plt.savefig('experiment_history_analysis.png', dpi=300, bbox_inches='tight')
            #plt.show()

    except Exception as e:
        print(f"历史数据分析失败: {e}")

    print("\n实验完成！所有数据已记录。")


    # --- 14. 添加快速查看最新结果的功能 ---
    def show_latest_results(csv_path='datacollection.csv', n_latest=3):
        """
        显示最近的实验结果
        """
        if not os.path.exists(csv_path):
            print("暂无实验数据")
            return

        df = pd.read_csv(csv_path)
        latest_timestamps = df['timestamp'].unique()[-n_latest:]

        print(f"\n{'=' * 80}")
        print(f"最近 {n_latest} 次实验结果")
        print(f"{'=' * 80}")

        for timestamp in latest_timestamps:
            timestamp_data = df[df['timestamp'] == timestamp]
            print(f"\n实验时间: {timestamp}")
            for _, row in timestamp_data.iterrows():
                print(f"  模型: {row['model_name']:15} | FID: {row.get('fid', 'N/A'):8.4f} | "
                      f"PSNR: {row.get('psnr', 'N/A'):8.4f} | 多样性: {row.get('diversity', 'N/A'):8.4f}")


    # 显示最新结果
    show_latest_results(csv_path)

    print(f"\n所有实验数据保存在: {csv_path}")
    print("可以使用 analyze_experiment_history() 函数随时分析历史数据")