import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# --- 1. 定义文件和实验名称 ---
LOG_FILE_PATH = "experiment_log_mnist.csv"
METHODS = [
    "Baseline (No Guide)",
    "Baseline (OT Guide)",
    "OT-Trained (No Guide)",
    "OT-Trained (OT Guide)"
]
METRICS_TO_PLOT = {
    "FID": {"lower_is_better": True},
    "KID": {"lower_is_better": True, "multiply_by": 1000.0, "display_name": "KID (x1e3)"},
    "SSIM": {"lower_is_better": False},
    "PSNR": {"lower_is_better": False, "display_name": "PSNR (dB)"},
    "Diversity (LPIPS)": {"lower_is_better": False}
}


# --- 2. 主绘图函数 ---
def plot_latest_experiment(log_file):
    print(f"正在读取日志文件: {log_file}")

    # --- 2.1 检查和读取数据 ---
    if not os.path.exists(log_file):
        print(f"错误: 未找到日志文件 '{log_file}'。")
        print("请先运行主实验脚本以生成日志。")
        return

    try:
        df = pd.read_csv(log_file)
        if df.empty:
            print("错误: 日志文件为空。")
            return
    except pd.errors.EmptyDataError:
        print("错误: 日志文件为空或格式不正确。")
        return
    except Exception as e:
        print(f"读取CSV时发生错误: {e}")
        return

    # 提取最后一次实验的数据
    latest_log = df.iloc[14]
    experiment_time = latest_log.get('timestamp', 'N/A')
    print(f"正在为 {experiment_time} 的实验绘制数据...")

    # --- 2.2 准备绘图 ---
    num_metrics = len(METRICS_TO_PLOT)
    fig, axes = plt.subplots(1, num_metrics, figsize=(24, 6))

    # 为4种方法设置固定颜色
    colors = plt.cm.get_cmap('Pastel1', 4)
    method_colors = [colors(i) for i in range(4)]

    # --- 2.3 循环绘制每个指标 ---
    for i, (metric_name, properties) in enumerate(METRICS_TO_PLOT.items()):
        ax = axes[i]
        values = []

        # 提取每种方法的值
        for method in METHODS:
            col_name = f"{method} - {metric_name}"
            if col_name not in latest_log:
                print(f"警告: 在日志中未找到列 '{col_name}'。")
                values.append(0)
                continue

            val = latest_log[col_name]

            # (例如 KLD x1e3)
            if "multiply_by" in properties:
                val *= properties["multiply_by"]
            values.append(val)

        # 绘制柱状图
        bars = ax.bar(METHODS, values, color=method_colors)

        # 添加数据标签
        ax.bar_label(bars, fmt='%.2f', padding=3)

        # --- 2.4 设置图表样式 ---
        display_name = properties.get("display_name", metric_name)
        if properties["lower_is_better"]:
            ax.set_title(f"{display_name}\n(The lower the better)", fontweight='bold')
            # 将最好的（最低的）柱子标为绿色
            best_val_idx = np.argmin(values)
            bars[best_val_idx].set_color('mediumseagreen')
        else:
            ax.set_title(f"{display_name}\n(The higher the better)", fontweight='bold')
            # 将最好的（最高的）柱子标为绿色
            best_val_idx = np.argmax(values)
            bars[best_val_idx].set_color('mediumseagreen')

        # 旋转X轴标签以便阅读
        ax.set_xticklabels(METHODS, rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # 自动调整Y轴范围
        min_val = min(values) * 0.9 if min(values) > 0 else min(values) * 1.1
        max_val = max(values) * 1.15  # 留出空间给标签
        ax.set_ylim([min_val, max_val])

    # --- 2.5 最终调整和保存 ---
    fig.suptitle(
        f"Comparison (N={latest_log.get('n_shot', '?')}, Class={latest_log.get('target_class', '?')})",
        fontsize=16, y=1.05)
    plt.tight_layout()

    save_filename = "latest_experiment_comparison.png"
    plt.savefig(save_filename, dpi=150, bbox_inches='tight')
    print(f"图表已保存到: {save_filename}")
    plt.show()


# --- 3. 运行脚本 ---
if __name__ == "__main__":
    # 设置 matplotlib 支持中文（用于标题）
    # 请根据您的操作系统选择合适的字体
    try:
        # 优先使用 'SimHei' (黑体)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    except:
        try:
            # 备选 'Microsoft YaHei' (微软雅黑)
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            print("警告：未找到中文字体 'SimHei' 或 'Microsoft YaHei'。图表中的中文可能显示为方框。")

    plot_latest_experiment(LOG_FILE_PATH)