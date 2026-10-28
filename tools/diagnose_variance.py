import os
import torch
import matplotlib.pyplot as plt
import numpy as np

def diagnose_action_variance(model, obs_batch, num_samples=32, save_dir='outputs/diagnose'):
    """
    诊断分析模型对于同一组输入的多次预测方差与轨迹平滑度。
    参数:
        model: 已加载好的并且处于 eval() 模式的 Policy 模型
        obs_batch: 一个 batch 的观测数据 (包含 head_cam, front_cam, agent_pos 等)
        num_samples: 对同一组数据重复采样的次数
        save_dir: 诊断图片的保存路径
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    print(f"开始进行 {num_samples} 次并行重复采样...")
    with torch.no_grad():
        with torch.autocast(device_type=str(model.device), dtype=torch.bfloat16):
            # 将同一观测输入模型，这里参考 eval.py 采用沿着 Batch 维度 repeat 的方式一次性并行采样
            sample_batch = {k: v[0:1].repeat(num_samples, *([1] * (v.dim() - 1))) for k, v in obs_batch.items()}
            res = model.predict_action(sample_batch)
            all_actions = res['action'].float().cpu().numpy()
        torch.cuda.empty_cache()
            
    num_samples, T_act, action_dim = all_actions.shape
    
    print(f"采样完成！轨迹形状: 采样数={num_samples}, Chunk步数={T_act}, 动作维度={action_dim}")
    
    # 绘制各维度的动作轨迹
    cols = 4
    rows = (action_dim + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    axes = axes.flatten()
    
    for dim in range(action_dim):
        ax = axes[dim]
        for s in range(num_samples):
            ax.plot(range(T_act), all_actions[s, :, dim], alpha=0.5)
        ax.set_title(f'Action Dim {dim}')
        ax.set_xlabel('Time Step (in Chunk)')
        ax.set_ylabel('Normalized Value')
        
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'action_variance_diagnosis.png')
    plt.savefig(save_path)
    print(f"诊断可视化图已保存至: {save_path}")
    
    # 计算诊断指标
    # 1. 采样间差异 (Variance across samples): 衡量对于同一个状态，模型生成不同动作的随机性
    sample_var = np.var(all_actions, axis=0).mean()
    
    # 2. 单条轨迹毛刺感 (Temporal smoothness): 衡量模型生成的单挑轨迹随着时间是否平滑
    diffs = np.diff(all_actions, axis=1) # [num_samples, T_act-1, action_dim]
    temporal_variance = np.var(diffs, axis=1).mean()
    
    print("-" * 50)
    print("【诊断指标结果】")
    print(f"1. 采样间不确定性 (Mean Sample Variance): {sample_var:.6f}")
    if sample_var > 0.05:
        print("  -> 分析: 采样间方差极大！这意味着相同观测下 Flow Matching 给出了区别巨大的动作（多模态/不确定）。执行时如果重规划频率高，将导致严重的不连贯和抖动。建议引入 CFG 或时序EMA滤波！")
    else:
        print("  -> 分析: 采样间方差可控。同一状态下的输出结果大体上是一致的。")

    print(f"2. 单条轨迹毛刺率 (Mean Temporal Variance within trajectory): {temporal_variance:.6f}")
    if temporal_variance > 0.05:
        print("  -> 分析: 单条轨迹毛刺极多！模型虽然用的是 Action Chunking，但是时间维度的平滑性完全没有学到，可能是求解 ODE 积分时 Euler 步数太少 (现为20) 或积分截断误差太大。建议增大推理步数或使用 DPM-Solver！")
    else:
        print("  -> 分析: 单条预测出的 Chunk 轨迹是相对平滑的。")
    print("-" * 50)
    
    return all_actions

# 示例：如何在你自己的推断脚本中调用这段代码
if __name__ == "__main__":
    print("This is a diagnostic tool module. Please import `diagnose_action_variance` and pass your instantiated model and a batch of observation to it.")
    print("Usage:")
    print("from tools.diagnose_variance import diagnose_action_variance")
    print("diagnose_action_variance(model, obs_batch, num_samples=10)")
