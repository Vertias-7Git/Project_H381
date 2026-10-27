"""
原型系统统一运行脚本
整合数据生成、模型训练、态势地图生成和可视化
一键运行完整实验流程
"""

import os
import sys
import json
import pickle
import numpy as np
import time

# 检查关键依赖
def check_dependencies():
    missing = []
    try:
        import torch
    except ImportError:
        missing.append('torch')
    try:
        import sklearn
    except ImportError:
        missing.append('scikit-learn')
    try:
        import scipy
    except ImportError:
        missing.append('scipy')
    try:
        import matplotlib
    except ImportError:
        missing.append('matplotlib')
    try:
        import docx
    except ImportError:
        missing.append('python-docx')

    if missing:
        print("=" * 60)
        print("检测到缺少以下依赖包，请先安装:")
        print(f"  pip install {' '.join(missing)}")
        print("=" * 60)
        return False
    return True


def run_full_pipeline():
    """运行完整实验管道"""
    import torch
    from data_generation import SyntheticDataGenerator
    from env_classifier import EnvironmentStateClassifier, SingleStationClassifier
    from radio_environment_map import RadioEnvironmentMap
    from visualization import Visualizer
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, confusion_matrix)

    output_dir = './results'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./data', exist_ok=True)

    print("=" * 70)
    print("  基于AI的低空无线环境认知与风险预测 - 原型系统")
    print("  AI-Based Low-Altitude Wireless Environment Perception")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[系统信息] 使用设备: {device}")
    if device == 'cuda':
        print(f"           GPU: {torch.cuda.get_device_name(0)}")

    # ============================================================
    # Stage 1: 数据仿真生成
    # ============================================================
    print("\n" + "-" * 70)
    print("[阶段1/5] 生成低空无线环境仿真数据集")
    print("-" * 70)

    t0 = time.time()
    data_gen = SyntheticDataGenerator(random_seed=42)
    dataset = data_gen.generate_dataset(
        n_samples_per_env=1000,
        n_base_stations=4,
        area_size=1000,
        output_dir='./data'
    )
    t1 = time.time()
    print(f"  用时: {t1-t0:.1f}s")

    train_samples = dataset['train']
    test_samples = dataset['test']

    # ============================================================
    # Stage 2: 模型训练
    # ============================================================
    print("\n" + "-" * 70)
    print("[阶段2/5] 训练环境状态识别模型")
    print("-" * 70)

    classifier = EnvironmentStateClassifier(device=device)
    history = classifier.train_model(
        train_samples=train_samples,
        test_samples=test_samples,
        epochs=60,
        batch_size=128,
        lr=1e-3,
        model_save_path='./models/env_classifier_best.pth'
    )
    t2 = time.time()
    print(f"  训练用时: {t2-t1:.1f}s")

    # 加载最佳模型
    classifier.model.load_state_dict(torch.load('./models/env_classifier_best.pth'))

    # ============================================================
    # Stage 3: 分类性能评估
    # ============================================================
    print("\n" + "-" * 70)
    print("[阶段3/5] 环境状态识别性能评估")
    print("-" * 70)

    test_results = classifier.predict_batch(test_samples, return_probs=True)

    y_true = [r['true_id'] for r in test_results]
    y_pred = [r['predicted_id'] for r in test_results]

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }

    # 每类指标
    env_names_key = ['normal', 'obstacle', 'interference', 'rain']
    env_names_cn = ['正常', '障碍遮挡', '同频干扰', '降雨扰动']

    for i, name in enumerate(env_names_key):
        try:
            metrics[f'precision_{name}'] = precision_score(y_true, y_pred, labels=[i], average=None, zero_division=0)[0]
            metrics[f'recall_{name}'] = recall_score(y_true, y_pred, labels=[i], average=None, zero_division=0)[0]
            metrics[f'f1_{name}'] = f1_score(y_true, y_pred, labels=[i], average=None, zero_division=0)[0]
        except:
            metrics[f'precision_{name}'] = 0
            metrics[f'recall_{name}'] = 0
            metrics[f'f1_{name}'] = 0

    print(f"\n  ╔══════════════════════════════════════════════════════════╗")
    print(f"  ║          环境状态识别核心指标                           ║")
    print(f"  ╠══════════════════════════════════════════════════════════╣")
    print(f"  ║  综合识别准确率     : {metrics['accuracy']*100:>6.2f}%    目标: ≥90%      ║")
    print(f"  ║  加权精确率         : {metrics['precision_weighted']*100:>6.2f}%    目标: ≥90%      ║")
    print(f"  ║  加权召回率         : {metrics['recall_weighted']*100:>6.2f}%    目标: ≥90%      ║")
    print(f"  ║  F1 Score (加权)    : {metrics['f1_weighted']:>7.4f}    目标: ≥0.88     ║")
    print(f"  ╚══════════════════════════════════════════════════════════╝")

    print(f"\n  --- 各类别详细指标 ---")
    for i, (key, cn) in enumerate(zip(env_names_key, env_names_cn)):
        print(f"  [{cn:>6s}] 精确率={metrics[f'precision_{key}']*100:>5.1f}%  "
              f"召回率={metrics[f'recall_{key}']*100:>5.1f}%  "
              f"F1={metrics[f'f1_{key}']:>5.3f}")

    # 混淆矩阵
    cm = np.array(metrics['confusion_matrix'])
    print(f"\n  --- 混淆矩阵 ---")
    print(f"  {'':>8s} {'正常':>6s} {'遮挡':>6s} {'干扰':>6s} {'降雨':>6s}")
    for i, name in enumerate(env_names_cn):
        row = ' '.join([f'{cm[i][j]:>6d}' for j in range(4)])
        print(f"  {name:>8s} {row}")

    # 保存
    with open(os.path.join(output_dir, 'classification_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # ============================================================
    # Stage 4: 态势地图与时空感知
    # ============================================================
    print("\n" + "-" * 70)
    print("[阶段4/5] 无线环境态势地图生成与时空感知评估")
    print("-" * 70)

    rem = RadioEnvironmentMap(
        area_size=(1000, 1000),
        grid_resolution=(50, 50),
        interpolator='rbf'
    )

    bs_positions = dataset['metadata']['base_stations']

    # 覆盖率计算
    print(f"\n  --- 基站感知覆盖率 ---")
    coverage_results = []
    for n_stations in [1, 2, 3, 4]:
        used_stations = bs_positions[:n_stations]
        coverage = rem.compute_coverage_rate(used_stations, coverage_radius=350)
        coverage_results.append({'n_stations': n_stations, 'coverage_rate': coverage})
        print(f"  {n_stations}个基站有效感知率: {coverage*100:.1f}%")

    # 地图精度评估
    print(f"\n  --- 态势地图重构精度 ---")
    np.random.seed(42)
    map_eval_samples = test_samples[:200]

    true_points = []
    true_labels = []
    for s in map_eval_samples:
        pos = s['metadata']['rx_pos']
        true_points.append([pos[0], pos[1]])
        true_labels.append(s['metadata']['env_state_id'])

    true_points = np.array(true_points)
    true_labels = np.array(true_labels)

    pred_results = classifier.predict_batch(map_eval_samples)
    pred_labels = np.array([r['predicted_id'] for r in pred_results])

    true_map = rem.generate_map(true_points.tolist(), true_labels)
    pred_map = rem.generate_map(true_points.tolist(), pred_labels)

    map_error = rem.compute_reconstruction_error(true_map.squeeze(), pred_map.squeeze())
    print(f"  地图MAE: {map_error['mae']:.4f}")
    print(f"  地图RMSE: {map_error['rmse']:.4f}")
    print(f"  空间重构误差: {map_error['spatial_error_m']:.2f} m")

    map_metrics = {
        'coverage_single_station': coverage_results[0]['coverage_rate'],
        'coverage_multi_station_2': coverage_results[1]['coverage_rate'],
        'coverage_multi_station_3': coverage_results[2]['coverage_rate'],
        'coverage_multi_station_4': coverage_results[3]['coverage_rate'],
        'coverage_target': 0.85,
        'coverage_base': 0.50,
        'map_mae': map_error['mae'],
        'map_rmse': map_error['rmse'],
        'spatial_reconstruction_error_m': map_error['spatial_error_m'],
        'los_error_target_m': 5,
        'nlos_error_target_m': 20,
    }

    with open(os.path.join(output_dir, 'map_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(map_metrics, f, ensure_ascii=False, indent=2)

    # ============================================================
    # Stage 5: 可视化输出
    # ============================================================
    print("\n" + "-" * 70)
    print("[阶段5/5] 生成可视化报告")
    print("-" * 70)

    viz = Visualizer(output_dir=output_dir)

    # 1. 训练曲线
    viz.plot_training_history(history)

    # 2. 混淆矩阵
    viz.plot_confusion_matrix(cm, metrics)

    # 3. 特征分布
    viz.plot_feature_distribution(test_samples)

    # 4. 性能仪表板
    viz.plot_performance_dashboard(metrics)

    # 5. 多基站对比
    viz.plot_multi_station_comparison(coverage_results)

    # 6. 态势地图
    viz.plot_environment_map(
        pred_map,
        observation_points=true_points.tolist(),
        station_positions=bs_positions,
        title='预测环境态势地图',
        save_path=os.path.join(output_dir, 'predicted_map.png')
    )
    viz.plot_environment_map(
        true_map,
        observation_points=true_points.tolist(),
        station_positions=bs_positions,
        title='真实环境态势地图',
        save_path=os.path.join(output_dir, 'true_map.png')
    )
    viz.plot_comparison_map(true_map, pred_map)

    t3 = time.time()
    print(f"  可视化用时: {t3-t2:.1f}s")

    # ============================================================
    # 最终报告
    # ============================================================
    print("\n" + "=" * 70)
    print("                    综合性能评估报告")
    print("=" * 70)

    report = {
        '项目名称': '基于AI的低空无线环境认知与风险预测',
        '原型版本': 'v1.0',
        '运行时间': time.strftime('%Y-%m-%d %H:%M:%S'),

        # 分类指标
        '综合识别准确率': f"{metrics['accuracy']*100:.2f}%",
        '目标_识别准确率': '≥ 90%',
        '状态_识别准确率': '达成' if metrics['accuracy'] >= 0.90 else '未达成(原型)',

        '加权召回率': f"{metrics['recall_weighted']*100:.2f}%",
        '目标_召回率': '≥ 90%',
        '状态_召回率': '达成' if metrics['recall_weighted'] >= 0.90 else '未达成(原型)',

        'F1_Score': f"{metrics['f1_weighted']:.4f}",
        '目标_F1': '≥ 0.88',
        '状态_F1': '达成' if metrics['f1_weighted'] >= 0.88 else '未达成(原型)',

        '正常类召回率(虚检相关)': f"{metrics.get('recall_normal', 0)*100:.2f}%",
        '目标_正常检测率': '≥ 90%',
        '状态_虚检': '达成' if metrics.get('recall_normal', 0) >= 0.90 else '未达成(原型)',

        # 时空指标
        '单站感知率': f"{coverage_results[0]['coverage_rate']*100:.1f}%",
        '四站感知率': f"{coverage_results[3]['coverage_rate']*100:.1f}%",
        '目标_多站感知率': '≥ 85%',
        '状态_感知率': '达成' if coverage_results[3]['coverage_rate'] >= 0.85 else '未达成(原型)',

        '地图重构空间误差': f"{map_error['spatial_error_m']:.2f} m",
        '目标_LOS误差': '≤ 5 m',
        '目标_NLOS误差': '≤ 20 m',
    }

    for key, value in report.items():
        if '目标' not in key and '状态' not in key:
            print(f"  {key:>20s}: {value}")

    print(f"\n  --- 与关键技术指标对比 ---")
    print(f"  {'指标':>20s} {'原型值':>12s} {'目标值':>12s} {'状态':>8s}")
    print(f"  {'-'*56}")
    checks = [
        ('识别准确率', f"{metrics['accuracy']*100:.2f}%", '≥ 90%', metrics['accuracy'] >= 0.90),
        ('召回率', f"{metrics['recall_weighted']*100:.2f}%", '≥ 90%', metrics['recall_weighted'] >= 0.90),
        ('F1 Score', f"{metrics['f1_weighted']:.4f}", '≥ 0.88', metrics['f1_weighted'] >= 0.88),
        ('多站感知率', f"{coverage_results[3]['coverage_rate']*100:.1f}%", '≥ 85%', coverage_results[3]['coverage_rate'] >= 0.85),
        ('空间误差', f"{map_error['spatial_error_m']:.2f}m", '≤ 20m', map_error['spatial_error_m'] <= 20),
    ]
    for name, val, target, ok in checks:
        status = '达成' if ok else '未达成'
        print(f"  {name:>20s} {val:>12s} {target:>12s} {status:>8s}")

    with open(os.path.join(output_dir, 'final_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"  实验完成! 总用时: {t3-t0:.1f}s")
    print(f"  结果目录: {os.path.abspath(output_dir)}")
    print(f"  输出文件:")
    for f in sorted(os.listdir(output_dir)):
        fp = os.path.join(output_dir, f)
        size = os.path.getsize(fp)
        print(f"    - {f} ({size/1024:.1f} KB)")
    print(f"{'='*70}")

    return {
        'metrics': metrics,
        'map_metrics': map_metrics,
        'coverage_results': coverage_results,
    }


if __name__ == '__main__':
    if not check_dependencies():
        sys.exit(1)

    results = run_full_pipeline()
