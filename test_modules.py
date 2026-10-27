"""
快速测试脚本 - 不依赖sklearn的简化版本
用于验证核心数据生成和模型结构是否正确
"""

import os
import sys
import numpy as np

# 测试数据生成
print("=" * 50)
print("测试1: 数据生成模块")
print("=" * 50)

from data_generation import SyntheticDataGenerator
gen = SyntheticDataGenerator(random_seed=42)

dataset = gen.generate_dataset(
    n_samples_per_env=200,  # 小规模快速测试
    n_base_stations=4,
    area_size=1000,
    output_dir='./data'
)

print(f"训练样本: {len(dataset['train'])}")
print(f"测试样本: {len(dataset['test'])}")

sample = dataset['train'][0]
print(f"\n样本特征示例:")
print(f"  RSS: {sample['features']['rss']:.2f} dBm")
print(f"  PathLoss: {sample['features']['path_loss']:.2f} dB")
print(f"  CSI实部长度: {len(sample['features']['csi_real'])}")
print(f"  环境: {sample['metadata']['env_state_name']}")
print(f"  频段: {sample['metadata']['band_name']}")

# 测试模型结构
print("\n" + "=" * 50)
print("测试2: 模型结构")
print("=" * 50)

import torch
from env_classifier import SingleStationClassifier, EnvironmentStateClassifier

model = SingleStationClassifier()
print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")

# 单样本前向传播测试
classifier = EnvironmentStateClassifier(model=model, device='cpu')
result = classifier.predict(sample)
print(f"\n单样本预测:")
print(f"  预测状态: {result['predicted_state']}")
print(f"  置信度: {result['confidence']:.4f}")

# 批量预测测试
batch_samples = dataset['test'][:10]
batch_results = classifier.predict_batch(batch_samples)
correct = sum(1 for r in batch_results if r['predicted_id'] == r['true_id'])
print(f"\n批量预测 (10样本):")
print(f"  正确: {correct}/{len(batch_results)}")

# 测试地图模块
print("\n" + "=" * 50)
print("测试3: 态势地图模块")
print("=" * 50)

from radio_environment_map import RadioEnvironmentMap

rem = RadioEnvironmentMap(area_size=(1000, 1000), grid_resolution=(30, 30))

points = [[s['metadata']['rx_pos'][0], s['metadata']['rx_pos'][1]] for s in dataset['test'][:30]]
values = np.array([s['metadata']['env_state_id'] for s in dataset['test'][:30]])

map_data = rem.generate_map(points, values)
print(f"地图形状: {map_data.shape}")

# 覆盖率
bs_pos = dataset['metadata']['base_stations']
coverage = rem.compute_coverage_rate(bs_pos, coverage_radius=350)
print(f"四基站覆盖率: {coverage*100:.1f}%")

print("\n" + "=" * 50)
print("所有基础模块测试通过!")
print("=" * 50)
print("\n要运行完整训练管道，请执行:")
print("  python run_prototype.py")
