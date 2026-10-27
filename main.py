"""
主运行脚本
整合数据生成、模型训练、态势地图生成和可视化
"""

import os
import sys
import json
import time
import pickle
import numpy as np
import torch

from data_generation import SyntheticDataGenerator
from env_classifier import EnvironmentStateClassifier, SingleStationClassifier
from radio_environment_map import RadioEnvironmentMap


def run_training_pipeline(output_dir='./results',
                          n_samples_per_env=1000,
                          epochs=50,
                          batch_size=64):
    """运行完整的训练管道"""

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./data', exist_ok=True)

    print("=" * 60)
    print("低空无线环境认知与风险预测 - 原型系统")
    print("=" * 60)

    # ============================================================
    # 阶段1: 生成仿真数据集
    # ============================================================
    print("\n[阶段1] 生成仿真数据集...")
    data_gen = SyntheticDataGenerator(random_seed=42)
    dataset = data_gen.generate_dataset(
        n_samples_per_env=n_samples_per_env,
        n_base_stations=4,
        area_size=1000,
        output_dir='./data'
    )

    train_samples = dataset['train']
    test_samples = dataset['test']

    # ============================================================
    # 阶段2: 训练环境状态识别模型
    # ============================================================
    print("\n[阶段2] 训练环境状态识别模型...")
    classifier = EnvironmentStateClassifier(device='cpu')
    history = classifier.train_model(
        train_samples=train_samples,
        test_samples=test_samples[:500],
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-3,
        model_save_path='./models/env_classifier.pth'
    )

    # 加载最佳模型
    classifier.model.load_state_dict(torch.load('./models/env_classifier.pth'))

    # ============================================================
    # 阶段3: 在测试集上评估
    # ============================================================
    print("\n[阶段3] 模型性能评估...")
    test_results = classifier.predict_batch(test_samples, return_probs=True)

    # 计算指标
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, confusion_matrix, classification_report)

    y_true = [r['true_id'] for r in test_results]
    y_pred = [r['predicted_id'] for r in test_results]

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }

    # 每类指标
    class_precision = precision_score(y_true, y_pred, average=None)
    class_recall = recall_score(y_true, y_pred, average=None)
    class_f1 = f1_score(y_true, y_pred, average=None)

    env_names = ['normal', 'obstacle', 'interference', 'rain']
    for i, name in enumerate(env_names):
        metrics[f'precision_{name}'] = class_precision[i]
        metrics[f'recall_{name}'] = class_recall[i]
        metrics[f'f1_{name}'] = class_f1[i]

    print(f"\n--- 核心分类指标 ---")
    print(f"综合识别准确率: {metrics['accuracy']*100:.2f}%")
    print(f"加权精确率: {metrics['precision_weighted']*100:.2f}%")
    print(f"加权召回率: {metrics['recall_weighted']*100:.2f}%")
    print(f"F1 Score (加权): {metrics['f1_weighted']:.4f}")
    print(f"\n--- 各类别指标 ---")
    for name in env_names:
        print(f"[{name:12s}] P={metrics[f'precision_{name}']*100:.1f}%  "
              f"R={metrics[f'recall_{name}']*100:.1f}%  "
              f"F1={metrics[f'f1_{name}']:.3f}")

    # 混淆矩阵
    cm = metrics['confusion_matrix']
    print(f"\n--- 混淆矩阵 ---")
    print(f"{'':>10s} {'正常':>6s} {'遮挡':>6s} {'干扰':>6s} {'降雨':>6s}")
    for i, name in enumerate(['正常', '遮挡', '干扰', '降雨']):
        row = ' '.join([f'{cm[i][j]:>6d}' for j in range(4)])
        print(f"{name:>10s} {row}")

    # 保存指标
    with open(os.path.join(output_dir, 'classification_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # ============================================================
    # 阶段4: 态势地图生成与评估
    # ============================================================
    print("\n[阶段4] 无线环境态势地图生成...")
    rem = RadioEnvironmentMap(
        area_size=(1000, 1000),
        grid_resolution=(50, 50),
        interpolator='rbf'
    )

    # 使用测试集中的观测点构建地图
    # 每个基站取一部分观测点
    bs_positions = dataset['metadata']['base_stations']

    coverage_results = []
    for n_stations in [1, 2, 3, 4]:
        used_stations = bs_positions[:n_stations]
        coverage = rem.compute_coverage_rate(used_stations, coverage_radius=350)
        coverage_results.append({
            'n_stations': n_stations,
            'coverage_rate': coverage,
        })
        print(f"  {n_stations}个基站有效感知率: {coverage*100:.1f}%")

    # 态势地图精度评估
    print("\n[阶段4.1] 态势地图精度评估...")

    # 构建真实地图和估计地图
    np.random.seed(42)
    map_eval_samples = test_samples[:200]

    # 收集各位置的真实标签
    true_points = []
    true_labels = []
    for s in map_eval_samples:
        pos = s['metadata']['rx_pos']
        true_points.append([pos[0], pos[1]])
        true_labels.append(s['metadata']['env_state_id'])

    true_points = np.array(true_points)
    true_labels = np.array(true_labels)

    # 用模型预测标签
    pred_results = classifier.predict_batch(map_eval_samples)
    pred_labels = np.array([r['predicted_id'] for r in pred_results])

    # 生成真实地图和估计地图
    true_map = rem.generate_map(true_points.tolist(), true_labels)
    pred_map = rem.generate_map(true_points.tolist(), pred_labels)

    # 计算重构误差
    map_error = rem.compute_reconstruction_error(true_map.squeeze(), pred_map.squeeze())
    print(f"  地图MAE: {map_error['mae']:.4f}")
    print(f"  地图MSE: {map_error['mse']:.4f}")
    print(f"  地图RMSE: {map_error['rmse']:.4f}")
    print(f"  空间重构误差: {map_error['spatial_error_m']:.2f} m")

    # 保存地图评估结果
    map_metrics = {
        'coverage_single_station': coverage_results[0]['coverage_rate'],
        'coverage_multi_station_2': coverage_results[1]['coverage_rate'],
        'coverage_multi_station_3': coverage_results[2]['coverage_rate'],
        'coverage_multi_station_4': coverage_results[3]['coverage_rate'],
        'map_mae': map_error['mae'],
        'map_rmse': map_error['rmse'],
        'spatial_reconstruction_error_m': map_error['spatial_error_m'],
    }

    with open(os.path.join(output_dir, 'map_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(map_metrics, f, ensure_ascii=False, indent=2)

    # ============================================================
    # 阶段5: 统计综合报告
    # ============================================================
    print("\n" + "=" * 60)
    print("综合性能报告")
    print("=" * 60)

    report = {
        '项目名称': '基于AI的低空无线环境认知与风险预测',
        '原型版本': 'v1.0',

        # 核心基础能力
        '核心识别准确率': f"{metrics['accuracy']*100:.2f}%",
        '目标核心识别准确率': '≥ 90%',
        '达成状态': '已达成' if metrics['accuracy'] >= 0.90 else '未达成(原型阶段)',

        '风险状态召回率_加权': f"{metrics['recall_weighted']*100:.2f}%",
        '目标风险召回率': '≥ 90%',
        '达成状态_召回': '已达成' if metrics['recall_weighted'] >= 0.90 else '未达成(原型阶段)',

        '虚检率_正常类特异度': f"{(1 - metrics['recall_normal'])*100:.2f}%",
        '目标虚检率': '≤ 10%',
        '达成状态_虚检': '已达成' if (1 - metrics['recall_normal']) <= 0.10 else '未达成(原型阶段)',

        'F1_Score': f"{metrics['f1_weighted']:.4f}",
        '目标F1': '≥ 0.88',
        '达成状态_F1': '已达成' if metrics['f1_weighted'] >= 0.88 else '未达成(原型阶段)',

        # 时空感知能力
        '单站感知率': f"{coverage_results[0]['coverage_rate']*100:.1f}%",
        '多站(4基站)感知率': f"{coverage_results[3]['coverage_rate']*100:.1f}%",
        '目标多站感知率': '≥ 85%',
        '达成状态_感知率': '已达成' if coverage_results[3]['coverage_rate'] >= 0.85 else '未达成(原型阶段)',

        '地图重构空间误差': f"{map_error['spatial_error_m']:.2f} m",
        '目标LOS误差': '≤ 5 m',
        '目标NLOS误差': '≤ 20 m',

        # 训练信息
        '训练样本数': len(train_samples),
        '测试样本数': len(test_samples),
        '训练轮数': epochs,
    }

    for key, value in report.items():
        print(f"  {key}: {value}")

    with open(os.path.join(output_dir, 'final_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"结果已保存至: {output_dir}")
    print("=" * 60)

    return {
        'metrics': metrics,
        'map_metrics': map_metrics,
        'report': report,
        'classifier': classifier,
        'rem': rem,
        'dataset': dataset,
    }


if __name__ == '__main__':
    # 检查依赖
    try:
        import sklearn
        import scipy
        import matplotlib
    except ImportError as e:
        print(f"警告: 缺少依赖包 {e}")
        print("请运行: pip install scikit-learn scipy matplotlib")
        sys.exit(1)

    results = run_training_pipeline(
        output_dir='./results',
        n_samples_per_env=1000,  # 原型阶段，可增大以提高精度
        epochs=50,
        batch_size=64
    )
