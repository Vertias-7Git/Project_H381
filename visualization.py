"""
可视化模块
为低空无线环境认知系统提供完整的可视化输出
包括：训练曲线、混淆矩阵、态势地图、性能仪表板等
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import LinearSegmentedColormap
import json
import os
from typing import Dict, List, Tuple


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Visualizer:
    """可视化工具类"""

    ENV_STATE_NAMES = ['正常', '障碍遮挡', '同频干扰', '降雨扰动']
    ENV_STATE_NAMES_EN = ['Normal', 'Obstacle', 'Interference', 'Rain']
    ENV_COLORS = ['#2ecc71', '#f39c12', '#e74c3c', '#3498db']

    def __init__(self, output_dir='./results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_training_history(self, history: Dict, save_path=None):
        """绘制训练历史曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 损失曲线
        ax1 = axes[0]
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='训练损失')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('训练损失曲线', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 准确率曲线
        ax2 = axes[1]
        ax2.plot(epochs, [a * 100 for a in history['val_acc']], 'g-', linewidth=2, label='验证准确率')
        ax2.plot(epochs, [f * 100 for f in history['val_f1']], 'r--', linewidth=2, label='验证F1分数')
        ax2.axhline(y=90, color='orange', linestyle=':', linewidth=1.5, label='目标线 (90%)')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('百分比 (%)', fontsize=12)
        ax2.set_title('验证性能曲线', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path = save_path or os.path.join(self.output_dir, 'training_history.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[可视化] 训练曲线已保存: {path}")
        return path

    def plot_confusion_matrix(self, cm: np.ndarray, metrics: Dict = None, save_path=None):
        """绘制混淆矩阵"""
        fig, ax = plt.subplots(figsize=(8, 7))

        # 归一化
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)

        # 添加数值标注
        thresh = cm_norm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)',
                        ha="center", va="center",
                        color="white" if cm_norm[i, j] > thresh else "black",
                        fontsize=10)

        ax.set_xticks(np.arange(len(self.ENV_STATE_NAMES)))
        ax.set_yticks(np.arange(len(self.ENV_STATE_NAMES)))
        ax.set_xticklabels(self.ENV_STATE_NAMES)
        ax.set_yticklabels(self.ENV_STATE_NAMES)
        ax.set_xlabel('预测标签', fontsize=12)
        ax.set_ylabel('真实标签', fontsize=12)

        title = '混淆矩阵（归一化）'
        if metrics:
            title += f"\nAcc={metrics['accuracy']*100:.1f}% | F1={metrics['f1_weighted']:.3f}"
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()

        path = save_path or os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[可视化] 混淆矩阵已保存: {path}")
        return path

    def plot_feature_distribution(self, samples: List[Dict], save_path=None):
        """绘制不同环境状态下的特征分布"""
        import pandas as pd

        # 提取特征
        features_data = []
        for s in samples:
            f = s['features']
            features_data.append({
                'RSS': f['rss'],
                'PathLoss': f['path_loss'],
                'DelaySpread': f['delay_spread'],
                'Doppler': f['doppler'],
                'SNR': f['snr'],
                '环境': self.ENV_STATE_NAMES[s['metadata']['env_state_id']]
            })

        df = pd.DataFrame(features_data)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        feature_names = ['RSS', 'PathLoss', 'DelaySpread', 'Doppler', 'SNR']
        units = ['dBm', 'dB', 'ns', 'Hz', 'dB']

        for idx, (feat_name, unit) in enumerate(zip(feature_names, units)):
            ax = axes[idx]

            for env_id, env_name in enumerate(self.ENV_STATE_NAMES):
                env_data = df[df['环境'] == env_name][feat_name]
                ax.hist(env_data, bins=30, alpha=0.5, label=env_name,
                        color=self.ENV_COLORS[env_id])

            ax.set_xlabel(f'{feat_name} ({unit})', fontsize=11)
            ax.set_ylabel('频数', fontsize=11)
            ax.set_title(f'{feat_name} 特征分布', fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2)

        # 隐藏最后一个子图
        axes[5].axis('off')

        plt.tight_layout()
        path = save_path or os.path.join(self.output_dir, 'feature_distribution.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[可视化] 特征分布已保存: {path}")
        return path

    def plot_environment_map(self,
                            map_data: np.ndarray,
                            observation_points: List[Tuple[float, float]] = None,
                            station_positions: List[Tuple[float, float]] = None,
                            area_size: Tuple[float, float] = (1000, 1000),
                            title: str = '低空无线环境态势地图',
                            save_path=None):
        """
        绘制环境态势地图
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        ny, nx = map_data.shape[:2]
        extent = [0, area_size[0], 0, area_size[1]]

        # 环境状态颜色映射
        if map_data.ndim == 3:
            map_2d = map_data[:, :, 0]
        else:
            map_2d = map_data

        # 创建自定义colormap
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(self.ENV_COLORS)

        # 量化到整数标签用于显示
        map_quantized = np.clip(map_2d.round(), 0, 3).astype(int)

        im = ax.imshow(map_quantized, origin='lower', extent=extent,
                       cmap=cmap, vmin=-0.5, vmax=3.5, interpolation='bilinear')

        # 添加观测点
        if observation_points:
            pts = np.array(observation_points)
            ax.scatter(pts[:, 0], pts[:, 1], c='white', s=15, alpha=0.3, marker='.')

        # 添加基站位置
        if station_positions:
            for i, pos in enumerate(station_positions):
                ax.scatter([pos[0]], [pos[1]], c='black', s=200, marker='^',
                          edgecolors='white', linewidths=2, zorder=5)
                ax.annotate(f'BS{i+1}', (pos[0], pos[1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, fontweight='bold', color='white')

                # 覆盖范围
                coverage = Circle((pos[0], pos[1]), 350, fill=False,
                                  edgecolor='white', linestyle='--',
                                  linewidth=1, alpha=0.5)
                ax.add_patch(coverage)

        # 图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=self.ENV_COLORS[i],
                                edgecolor='black',
                                label=self.ENV_STATE_NAMES[i])
                          for i in range(4)]
        ax.legend(handles=legend_elements, loc='upper left',
                 fontsize=10, title='环境状态')

        ax.set_xlabel('X 位置 (m)', fontsize=12)
        ax.set_ylabel('Y 位置 (m)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()
        path = save_path or os.path.join(self.output_dir, 'environment_map.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[可视化] 态势地图已保存: {path}")
        return path

    def plot_performance_dashboard(self, metrics: Dict, save_path=None):
        """
        绘制性能仪表板 - 综合展示所有关键指标
        """
        fig = plt.figure(figsize=(16, 12))

        # 创建子图布局
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        # === 1. 核心指标仪表盘 ===
        ax1 = fig.add_subplot(gs[0, 0])
        self._draw_gauge(ax1, metrics['accuracy'] * 100, 100, '识别准确率',
                        target=90, unit='%')

        ax2 = fig.add_subplot(gs[0, 1])
        self._draw_gauge(ax2, metrics['recall_weighted'] * 100, 100, '风险召回率',
                        target=90, unit='%')

        ax3 = fig.add_subplot(gs[0, 2])
        false_alarm = (1 - metrics.get('recall_normal', metrics.get('precision_normal', 0.9))) * 100
        self._draw_gauge(ax3, 100 - false_alarm, 100, '正常检测率',
                        target=90, unit='%', inverse=True)

        # === 2. F1分数柱状图 ===
        ax4 = fig.add_subplot(gs[1, 0])
        env_names = ['正常', '遮挡', '干扰', '降雨']
        f1_scores = [metrics.get(f'f1_{n}', 0) for n in ['normal', 'obstacle', 'interference', 'rain']]
        recalls = [metrics.get(f'recall_{n}', 0) for n in ['normal', 'obstacle', 'interference', 'rain']]

        x = np.arange(len(env_names))
        width = 0.35
        bars1 = ax4.bar(x - width/2, f1_scores, width, label='F1 Score', color='#3498db')
        bars2 = ax4.bar(x + width/2, recalls, width, label='Recall', color='#e74c3c')

        ax4.axhline(y=0.88, color='orange', linestyle='--', linewidth=1.5, label='目标F1 (0.88)')
        ax4.axhline(y=0.90, color='green', linestyle='--', linewidth=1.5, label='目标Recall (0.90)')
        ax4.set_ylabel('Score', fontsize=11)
        ax4.set_title('各类环境状态性能对比', fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(env_names)
        ax4.legend(fontsize=8)
        ax4.set_ylim(0, 1.1)
        ax4.grid(True, alpha=0.2)

        # 添加数值标注
        for bar in bars1:
            height = bar.get_height()
            ax4.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

        # === 3. 精确率对比 ===
        ax5 = fig.add_subplot(gs[1, 1])
        precisions = [metrics.get(f'precision_{n}', 0) for n in ['normal', 'obstacle', 'interference', 'rain']]
        colors = ['#2ecc71', '#f39c12', '#e74c3c', '#3498db']
        wedges, texts, autotexts = ax5.pie(precisions, labels=env_names, colors=colors,
                                            autopct='%1.1f%%', startangle=90)
        ax5.set_title('各类别精确率占比', fontsize=12, fontweight='bold')

        # === 4. 综合指标表格 ===
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')

        table_data = [
            ['指标', '原型值', '目标值', '差距'],
            ['识别准确率', f'{metrics["accuracy"]*100:.2f}%', '≥90%', self._status(metrics['accuracy'], 0.90, True)],
            ['加权召回率', f'{metrics["recall_weighted"]*100:.2f}%', '≥90%', self._status(metrics['recall_weighted'], 0.90, True)],
            ['加权F1', f'{metrics["f1_weighted"]:.4f}', '≥0.88', self._status(metrics['f1_weighted'], 0.88, True)],
            ['正常检测率', f'{metrics.get("recall_normal", 0)*100:.2f}%', '≥90%', self._status(metrics.get('recall_normal', 0), 0.90, True)],
        ]

        table = ax6.table(cellText=table_data[1:], colLabels=table_data[0],
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)

        # 设置表头颜色
        for i in range(4):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(color='white', fontweight='bold')

        ax6.set_title('核心指标对比表', fontsize=12, fontweight='bold', pad=20)

        # === 5. 项目标题 ===
        fig.suptitle('低空无线环境认知原型系统 - 性能仪表板',
                    fontsize=16, fontweight='bold', y=0.98)

        path = save_path or os.path.join(self.output_dir, 'performance_dashboard.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[可视化] 性能仪表板已保存: {path}")
        return path

    def _draw_gauge(self, ax, value, max_value, title, target=None, unit='',
                   inverse=False):
        """绘制半圆仪表盘"""
        theta = np.linspace(0, np.pi, 100)
        r = 1.0

        # 背景弧
        ax.fill_between(np.cos(theta), np.sin(theta), 0, alpha=0.1, color='gray')

        # 值弧
        value_theta = theta[int(value / max_value * 99)]
        value_arc = theta[:int(value / max_value * 99) + 1]
        ax.fill_between(np.cos(value_arc) * r, np.sin(value_arc) * r, 0,
                        alpha=0.6, color='#3498db')

        # 目标线
        if target is not None:
            target_theta = theta[int(target / max_value * 99)]
            ax.plot([0, np.cos(target_theta) * 0.9],
                   [0, np.sin(target_theta) * 0.9],
                   'r--', linewidth=2, label=f'目标: {target}{unit}')

        # 指针
        ax.plot([0, np.cos(value_theta) * 0.85],
               [0, np.sin(value_theta) * 0.85],
               'k-', linewidth=3)
        ax.scatter([0], [0], c='black', s=50, zorder=5)

        # 文字
        ax.text(0, -0.3, f'{value:.2f}{unit}', ha='center', fontsize=18, fontweight='bold')
        ax.text(0, -0.55, title, ha='center', fontsize=11)

        if target is not None:
            target_label = '达标' if (value >= target if not inverse else value <= target) else '未达标'
            color = 'green' if (value >= target if not inverse else value <= target) else 'red'
            ax.text(0, -0.75, target_label, ha='center', fontsize=11, color=color, fontweight='bold')

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.0, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')

    def _status(self, value, target, higher_better=True):
        """返回指标状态字符串"""
        if higher_better:
            if value >= target:
                return '✓ 达成'
            elif value >= target * 0.5:
                return '△ 接近'
            else:
                return '✗ 未达成'
        else:
            if value <= target:
                return '✓ 达成'
            elif value <= target * 1.5:
                return '△ 接近'
            else:
                return '✗ 未达成'

    def plot_multi_station_comparison(self,
                                      coverage_results: List[Dict],
                                      save_path=None):
        """绘制多基站感知率对比"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 柱状图
        ax1 = axes[0]
        n_stations = [r['n_stations'] for r in coverage_results]
        rates = [r['coverage_rate'] * 100 for r in coverage_results]

        bars = ax1.bar([f'{n}站' for n in n_stations], rates,
                      color=['#e74c3c' if n == 1 else '#3498db' if n < 4 else '#2ecc71'
                             for n in n_stations],
                      edgecolor='black', linewidth=1.5)

        ax1.axhline(y=85, color='orange', linestyle='--', linewidth=2, label='目标: 85%')
        ax1.axhline(y=50, color='red', linestyle=':', linewidth=1.5, label='单站基线: ~50%')
        ax1.set_ylabel('有效感知率 (%)', fontsize=12)
        ax1.set_title('多基站协同感知率', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 100)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.2, axis='y')

        for bar, rate in zip(bars, rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{rate:.1f}%', ha='center', fontsize=12, fontweight='bold')

        # 对比表格
        ax2 = axes[1]
        ax2.axis('off')

        table_data = [
            ['基站数', '感知率', '相对基线提升', '目标达成'],
        ]
        base_rate = coverage_results[0]['coverage_rate']
        for r in coverage_results:
            n = r['n_stations']
            rate = r['coverage_rate']
            improvement = (rate - base_rate) / base_rate * 100 if base_rate > 0 else 0
            status = '✓' if rate >= 0.85 else '△' if rate >= 0.425 else '✗'
            table_data.append([
                f'{n}个基站',
                f'{rate*100:.1f}%',
                f'+{improvement:.1f}%' if n > 1 else '基线',
                status
            ])

        table = ax2.table(cellText=table_data[1:], colLabels=table_data[0],
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.3, 2)

        for i in range(4):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(color='white', fontweight='bold')

        ax2.set_title('多站感知性能概览', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        path = save_path or os.path.join(self.output_dir, 'multi_station_comparison.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[可视化] 多站对比已保存: {path}")
        return path

    def plot_comparison_map(self,
                           true_map: np.ndarray,
                           pred_map: np.ndarray,
                           area_size: Tuple[float, float] = (1000, 1000),
                           save_path=None):
        """绘制真实地图和预测地图对比"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(self.ENV_COLORS)

        # 真值地图
        if true_map.ndim == 3:
            true_map = true_map[:, :, 0]
        if pred_map.ndim == 3:
            pred_map = pred_map[:, :, 0]

        true_q = np.clip(true_map.round(), 0, 3).astype(int)
        pred_q = np.clip(pred_map.round(), 0, 3).astype(int)
        diff = np.abs(true_q - pred_q)

        titles = ['真实环境地图', '模型预测地图', '预测误差']
        datas = [true_q, pred_q, diff]
        cmaps = [cmap, cmap, plt.cm.Reds]
        vmins = [-0.5, -0.5, 0]
        vmaxs = [3.5, 3.5, 3]

        for ax, data, title, cm, vm, vx in zip(axes, datas, titles, cmaps, vmins, vmaxs):
            im = ax.imshow(data, origin='lower',
                          extent=[0, area_size[0], 0, area_size[1]],
                          cmap=cm, vmin=vm, vmax=vx, interpolation='bilinear')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('X (m)', fontsize=10)
            ax.set_ylabel('Y (m)', fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        path = save_path or os.path.join(self.output_dir, 'map_comparison.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[可视化] 地图对比已保存: {path}")
        return path

    def generate_all_visualizations(self,
                                    classifier_results,
                                    train_history,
                                    map_results,
                                    dataset):
        """生成所有可视化图表"""
        print("\n[可视化] 开始生成所有可视化结果...")

        metrics = classifier_results['metrics']
        test_samples = classifier_results.get('test_samples', [])

        # 1. 训练曲线
        if train_history:
            self.plot_training_history(train_history)

        # 2. 混淆矩阵
        cm = np.array(metrics['confusion_matrix'])
        self.plot_confusion_matrix(cm, metrics)

        # 3. 特征分布
        if test_samples:
            self.plot_feature_distribution(test_samples)

        # 4. 性能仪表板
        self.plot_performance_dashboard(metrics)

        # 5. 多基站对比
        if 'coverage_results' in map_results:
            self.plot_multi_station_comparison(map_results['coverage_results'])

        # 6. 态势地图
        if 'true_map' in map_results and 'pred_map' in map_results:
            self.plot_environment_map(
                map_results['pred_map'],
                station_positions=dataset['metadata']['base_stations'],
                title='预测环境态势地图',
                save_path=os.path.join(self.output_dir, 'predicted_map.png')
            )
            self.plot_environment_map(
                map_results['true_map'],
                station_positions=dataset['metadata']['base_stations'],
                title='真实环境态势地图',
                save_path=os.path.join(self.output_dir, 'true_map.png')
            )
            self.plot_comparison_map(
                map_results['true_map'],
                map_results['pred_map']
            )

        print("[可视化] 所有可视化结果生成完毕!")


if __name__ == '__main__':
    # 测试可视化模块
    viz = Visualizer(output_dir='./test_viz')

    # 测试仪表盘
    test_metrics = {
        'accuracy': 0.92,
        'precision_weighted': 0.91,
        'recall_weighted': 0.90,
        'f1_weighted': 0.905,
        'recall_normal': 0.95,
        'precision_normal': 0.93,
        'recall_obstacle': 0.88,
        'precision_obstacle': 0.87,
        'recall_interference': 0.89,
        'precision_interference': 0.90,
        'recall_rain': 0.85,
        'precision_rain': 0.88,
        'f1_normal': 0.94,
        'f1_obstacle': 0.875,
        'f1_interference': 0.895,
        'f1_rain': 0.865,
    }
    viz.plot_performance_dashboard(test_metrics)
    print("测试可视化完成!")
