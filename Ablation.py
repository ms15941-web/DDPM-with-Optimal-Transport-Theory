import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 1. [用户配置] ---
# ----------------------------
# 要读取的日志文件
LOG_FILE_PATH = "experiment_log_mnist.csv"

# 您想分析最后 k 次实验 (k 行)
# 例如，如果您为 n_shot=[10, 25, 50, 100] 运行了4次，请设置为 4
K_LAST_ROWS = 4

# 您想作为 X 轴的超参数 (必须与日志中的列名完全匹配)
# 例如: "n_shot", "ot_guidance_weight", "num_epochs"
CONDITION_PARAM = "n_shot"

# 您想作为 Y 轴的指标名称
# 例如: "FID", "KID", "SSIM", "PSNR", "Diversity (LPIPS)"
METRIC_NAME = "FID"
# ----------------------------

# --- 2. 脚本内部定义 ---
METHODS = [
    "Baseline (No Guide)",
    "Baseline (OT Guide)",
    "OT-Trained (No Guide)",
    "OT-Trained (OT Guide)"
]

# 用于美化图表标题
METRICS_PROPERTIES = {
    "FID": {"lower_is_better": True},
    "KID": {"lower_is_better": True, "multiply_by": 1000.0, "display_name": "KID (x1e3)"},
    "SSIM": {"lower_is_better": False},
    "PSNR": {"lower_is_better": False, "display_name": "PSNR (dB)"},
    "Diversity (LPIPS)": {"lower_is_better": False}
}


def plot_ablation_study(log_file, k_rows, condition_param, metric_name):
    print(f"开始绘制消融实验图表...")
    print(f"X轴 (Condition): {condition_param}")
    print(f"Y轴 (Metric): {metric_name}")
    print(f"数据点 (K-Rows): {k_rows}")

    # --- 2.1 检查和读取数据 ---
    if not os.path.exists(log_file):
        print(f"错误: 未找到日志文件 '{log_file}'。")
        return

    try:
        df = pd.read_csv(log_file)
        if df.empty:
            print("错误: 日志文件为空。")
            return
    except Exception as e:
        print(f"读取CSV时发生错误: {e}")
        return

    # --- 2.2 提取数据 ---
    if len(df) < k_rows:
        print(f"警告: 您要求 {k_rows} 行, 但日志中只有 {len(df)} 行。将使用所有行。")
        ablation_df = df
    else:
        # 提取最后 k 行数据
        ablation_df = df.tail(k_rows)

    # 按 X 轴参数排序，确保折线图正确
    try:
        ablation_df = ablation_df.sort_values(by=condition_param)
    except KeyError:
        print(f"错误: 在日志中未找到超参数列 '{condition_param}'。")
        print(f"可用的超参数列: {[col for col in df.columns if '-' not in col]}")
        return

    x_values = ablation_df[condition_param]
    plot_data = {}

    # 提取每种方法的 Y 轴数据
    for method in METHODS:
        col_name = f"{method} - {metric_name}"
        if col_name not in ablation_df.columns:
            print(f"警告: 在日志中未找到指标列 '{col_name}'。将跳过此方法。")
            continue

        y_values = ablation_df[col_name]

        # 检查是否需要缩放 (例如 KID)
        props = METRICS_PROPERTIES.get(metric_name, {})
        if "multiply_by" in props:
            y_values = y_values * props["multiply_by"]

        plot_data[method] = y_values

    if not plot_data:
        print(f"错误: 未能为指标 '{metric_name}' 找到任何有效数据。")
        print(f"请检查您的 METRIC_NAME 是否正确。")
        return

    # --- 2.3 绘制图表 ---
    plt.figure(figsize=(10, 6))

    for method_name, y_values in plot_data.items():
        plt.plot(x_values, y_values, marker='o', linestyle='-', label=method_name)

    # --- 2.4 设置图表样式 ---
    props = METRICS_PROPERTIES.get(metric_name, {})
    display_name = props.get("display_name", metric_name)
    direction = "(The lower the better)" if props.get("lower_is_better", False) else "(越高越好)"

    plt.title(f"Ablation: {display_name} vs. {condition_param}", fontsize=16)
    plt.xlabel(condition_param, fontsize=12)
    plt.ylabel(f"{display_name} {direction}", fontsize=12)

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # 确保 X 轴显示所有条件值
    plt.xticks(x_values)

    # --- 2.5 最终调整和保存 ---
    plt.tight_layout()

    save_filename = f"ablation_{metric_name}_vs_{condition_param}.png"
    plt.savefig(save_filename, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存到: {save_filename}")
    plt.show()


# --- 3. 运行脚本 ---
if __name__ == "__main__":
    # 设置 matplotlib 支持中文（用于标题）
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        try:
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            print("警告：未找到中文字体 'SimHei' 或 'Microsoft YaHei'。图表中的中文可能显示为方框。")

    # 运行主函数
    plot_ablation_study(LOG_FILE_PATH, K_LAST_ROWS, CONDITION_PARAM, METRIC_NAME)