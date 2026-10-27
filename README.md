# 基于AI的低空无线环境认知与风险预测 - 原型系统

## 项目简介

本原型系统对应大创项目 **"基于AI的低空无线环境认知与风险预测"**（北京邮电大学，网络空间安全学院）。

系统模拟了多基站、多频段的无线环境感知场景，通过深度学习实现环境状态智能识别，并利用空间插值生成连续的环境态势地图，最终达成"感知→识别→推理→预测"的完整技术链条。

## 技术路线

```
多频多站特征感知 
    → 环境状态智能识别 
    → 区域实时地图构建 
    → 动态时空趋势预测
```

## 文件结构

```
low_altitude_perception/
├── data_generation.py        # 仿真数据集生成（物理传播模型）
├── env_classifier.py         # 环境状态识别模型（PyTorch深度学习）
│   └── SingleStationClassifier    # 单基站分类器
│   └── MultiStationFusion         # 多基站联合感知
│   └── EnvironmentStateClassifier # 封装训练和推理
├── radio_environment_map.py  # 无线环境态势地图生成
│   └── SpatialInterpolator         # 空间插值（IDW/RBF/Kriging）
│   └── RadioEnvironmentMap         # 地图生成与评估
├── visualization.py          # 可视化输出模块
│   └── Visualizer                   # 训练曲线/混淆矩阵/态势地图/性能仪表板
├── run_prototype.py          # 主运行脚本（一键执行完整实验）
├── main.py                   # 简化的主函数入口
├── data/                     # 数据集输出目录
├── models/                   # 模型保存目录
└── results/                  # 实验结果与可视化输出
```

## 环境依赖

```bash
pip install torch numpy scikit-learn scipy matplotlib python-docx
```

## 运行方式

```bash
cd low_altitude_perception
python run_prototype.py
```

运行后会在 `./results/` 目录下生成所有可视化图表和JSON指标文件。

## 关键技术指标对照

| 指标 | 申请书目标 | 原型可接受范围 | 说明 |
|-----|-----------|--------------|------|
| 综合识别准确率 | ≥ 90% | ≥ 45% (±50%) | 核心分类性能 |
| 风险状态召回率 | ≥ 90% | ≥ 45% (±50%) | 对风险状态的检测能力 |
| 虚检率 | ≤ 10% | ≤ 15% | 正常环境的误报控制 |
| F1 Score | ≥ 0.88 | ≥ 0.44 (±50%) | 综合分类性能 |
| 多站感知率 | ≥ 85% | ≥ 42.5% (±50%) | 空间覆盖能力 |
| 态势地图更新频率 | 0.5-2 Hz | 视计算性能 | 实时性 |
| LOS空间误差 | ≤ 5m | ≤ 7.5m (±50%) | 地图重构精度 |
| NLOS空间误差 | ≤ 20m | ≤ 30m (±50%) | 地图重构精度 |
| 边缘处理时延 | < 60ms | 原型环境暂不支持 | 推理速度 |
| 跨场景衰减 | ≤ 15% | 原型环境暂不支持 | 泛化能力 |

## 输出文件

运行后 `results/` 目录包含：

- `training_history.png` - 训练损失与验证准确率曲线
- `confusion_matrix.png` - 混淆矩阵
- `feature_distribution.png` - 各环境状态下的特征分布
- `performance_dashboard.png` - 综合性能仪表板
- `multi_station_comparison.png` - 多基站感知率对比
- `predicted_map.png` / `true_map.png` - 态势地图
- `map_comparison.png` - 真值与预测地图对比
- `classification_metrics.json` - 分类指标JSON
- `map_metrics.json` - 地图评估指标JSON
- `final_report.json` - 最终综合报告

## 模型架构

### 单基站分类器
- CSI编码器：Conv1D提取信道状态信息特征
- 频段嵌入：3种频段的可学习嵌入
- 特征融合：RSS + CSI + PathLoss + DelaySpread + Doppler + SNR + RSRP + BandEmbed
- 分类头：MLP → 4类环境状态

### 多基站融合（扩展模块）
- 各基站特征独立提取
- 跨站注意力机制（MultiHeadAttention）
- 空间位置编码
- 联合决策输出

## 仿真数据特性

数据集包含4种环境状态：
1. **正常环境** - 视距传播，轻微随机衰落
2. **障碍遮挡** - 建筑物/树木穿透损耗，时延扩展增大
3. **同频干扰** - 电磁干扰导致SINR下降
4. **降雨扰动** - 雨衰影响（毫米波频段更敏感）

无线特征：RSSI、CSI(16-tap)、PathLoss、Delay Spread、Doppler、SNR、RSRP

频段：Sub-6GHz (3.5GHz)、28GHz毫米波、60GHz毫米波

## 开发者

北京邮电大学 网络空间安全学院 2024级
- 阮辅强（项目负责人）
- 寸羽萱
- 张若斐
- 陈甜甜
- 刘晨阳

指导教师：王申 副教授
