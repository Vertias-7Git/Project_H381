"""
数据仿真生成模块
基于物理传播模型，模拟多基站、多频段的无线信号特征数据
生成包含不同环境状态（正常/障碍遮挡/同频干扰/降雨扰动）的仿真数据集
"""

import numpy as np
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Tuple
import pickle


@dataclass
class WirelessFeature:
    """无线观测特征"""
    rss: float          # 接收信号强度 (dBm)
    csi_real: List[float]      # CSI实部（信道状态信息）
    csi_imag: List[float]      # CSI虚部
    path_loss: float    # 路径损耗 (dB)
    delay_spread: float # 时延扩展 (ns)
    doppler: float      # 多普勒频移 (Hz)
    snr: float          # 信噪比 (dB)
    rsrp: float         # 参考信号接收功率

@dataclass
class EnvironmentLabel:
    """环境状态标签"""
    label: str          # 'normal', 'obstacle', 'interference', 'rain'
    label_id: int       # 0, 1, 2, 3
    description: str


class PropagationModel:
    """无线传播模型基类"""
    @staticmethod
    def free_space_path_loss(frequency_mhz, distance_m):
        """自由空间路径损耗（Friis公式）"""
        c = 3e8  # 光速
        wavelength = c / (frequency_mhz * 1e6)
        if distance_m <= 0:
            distance_m = 1
        pl_db = 20 * np.log10(distance_m) + 20 * np.log10(frequency_mhz) - 147.55
        return pl_db

    @staticmethod
    def okumura_hata_path_loss(frequency_mhz, distance_m, hb=30, hm=10, environment='urban'):
        """Okumura-Hata 城市传播模型"""
        if distance_m < 1000:
            distance_km = distance_m / 1000
        else:
            distance_km = distance_m / 1000

        # 修正因子
        if frequency_mhz <= 1500:
            a_hm = (1.1 * np.log10(frequency_mhz) - 0.7) * hm - (1.56 * np.log10(frequency_mhz) - 0.8)
        else:
            a_hm = 3.2 * (np.log10(11.75 * hm))**2 - 4.97

        pl_db = 69.55 + 26.16 * np.log10(frequency_mhz) - 13.82 * np.log10(hb) - a_hm \
                + (44.9 - 6.55 * np.log10(hb)) * np.log10(distance_km)
        return pl_db

    @staticmethod
    def rain_attenuation(frequency_mhz, rain_rate_mm_h, distance_m):
        """降雨衰减模型（ITU-R P.838）"""
        # 简化模型
        if rain_rate_mm_h <= 0:
            return 0.0

        # 频率相关衰减系数
        if frequency_mhz < 10000:
            # 低频段（Sub-10GHz）降雨影响较小
            k = 0.0002 * (frequency_mhz / 1000)
            alpha = 0.8
        else:
            # 毫米波频段降雨影响较大
            k = 0.01 * (frequency_mhz / 1000)
            alpha = 1.2

        gamma = k * (rain_rate_mm_h ** alpha)
        attenuation_db = gamma * (distance_m / 1000)
        return attenuation_db

    @staticmethod
    def shadowing_fading(std_db=8.0):
        """阴影衰落（对数正态分布）"""
        return np.random.normal(0, std_db)

    @staticmethod
    def multipath_fading(n_taps=16):
        """多径衰落 - 生成CSI-like特征"""
        real = np.random.randn(n_taps) * 0.5
        imag = np.random.randn(n_taps) * 0.5
        return real.tolist(), imag.tolist()


class SyntheticDataGenerator:
    """仿真数据生成器"""

    # 频段配置
    BAND_CONFIG = {
        'sub_6ghz': {'band_name': 'Sub-6GHz', 'freq_mhz': 3500, 'wavelength_m': 0.0857},
        'mmwave_28': {'band_name': '28GHz毫米波', 'freq_mhz': 28000, 'wavelength_m': 0.0107},
        'mmwave_60': {'band_name': '60GHz毫米波', 'freq_mhz': 60000, 'wavelength_m': 0.005},
    }

    # 环境状态定义
    ENV_STATES = {
        0: {'name': 'normal', 'description': '正常环境（视距传播）'},
        1: {'name': 'obstacle', 'description': '障碍遮挡（建筑物/树木阻挡）'},
        2: {'name': 'interference', 'description': '同频干扰（电磁干扰）'},
        3: {'name': 'rain', 'description': '降雨扰动（雨衰影响）'},
    }

    def __init__(self, random_seed=42):
        np.random.seed(random_seed)
        self.prop_model = PropagationModel()

    def generate_base_station_layout(self, n_stations=4, area_size=1000):
        """
        生成基站布局
        area_size: 区域边长（米）
        返回: List of (x, y, height) tuples
        """
        # 在区域四角和中部部署基站
        positions = []
        if n_stations >= 4:
            # 四个角落的基站
            positions.append((0.1 * area_size, 0.1 * area_size, 30))  # 左下角
            positions.append((0.9 * area_size, 0.1 * area_size, 30))  # 右下角
            positions.append((0.1 * area_size, 0.9 * area_size, 35))  # 左上角
            positions.append((0.9 * area_size, 0.9 * area_size, 35))  # 右上角

        # 中心基站
        if n_stations >= 5:
            positions.append((0.5 * area_size, 0.5 * area_size, 50))

        # 额外基站
        for i in range(len(positions), n_stations):
            x = np.random.uniform(0.1, 0.9) * area_size
            y = np.random.uniform(0.1, 0.9) * area_size
            h = np.random.uniform(25, 50)
            positions.append((x, y, h))

        return positions[:n_stations]

    def generate_single_sample(self,
                               tx_pos, rx_pos,
                               band_name,
                               env_state_id: int,
                               tx_power_dbm=23,
                               antenna_gain_db=15) -> dict:
        """
        生成单个无线观测样本

        tx_pos: (x, y, z) 发射端位置
        rx_pos: (x, y, z) 接收端位置
        band_name: 频段名称
        env_state_id: 环境状态ID
        """
        band = self.BAND_CONFIG[band_name]
        freq_mhz = band['freq_mhz']

        # 计算距离
        distance_m = np.sqrt((tx_pos[0] - rx_pos[0])**2 +
                             (tx_pos[1] - rx_pos[1])**2 +
                             (tx_pos[2] - rx_pos[2])**2)

        # 基础路径损耗
        if distance_m < 100:
            pl_db = self.prop_model.free_space_path_loss(freq_mhz, distance_m)
        else:
            pl_db = self.prop_model.okumura_hata_path_loss(freq_mhz, distance_m)

        # 根据环境状态添加额外损耗和特征变化
        env_config = self.ENV_STATES[env_state_id]
        env_name = env_config['name']

        extra_loss = 0
        snr_degradation = 0
        delay_spread_factor = 1.0
        doppler_factor = 1.0
        csi_distortion = 1.0

        if env_name == 'normal':
            # 正常环境 - 视距
            extra_loss = np.random.normal(0, 2)
            snr_degradation = np.random.normal(0, 1)
            delay_spread_factor = np.random.uniform(1.0, 2.0)

        elif env_name == 'obstacle':
            # 障碍遮挡 - 穿透损耗
            penetration_loss = np.random.uniform(10, 30)
            extra_loss = penetration_loss + np.random.normal(0, 3)
            snr_degradation = penetration_loss * 0.8 + np.random.normal(0, 2)
            delay_spread_factor = np.random.uniform(3.0, 8.0)
            doppler_factor = np.random.uniform(0.5, 1.5)
            csi_distortion = np.random.uniform(1.5, 3.0)

        elif env_name == 'interference':
            # 同频干扰 - SINR下降
            interference_power = np.random.uniform(5, 20)
            extra_loss = interference_power * 0.3 + np.random.normal(0, 2)
            snr_degradation = interference_power + np.random.normal(0, 3)
            delay_spread_factor = np.random.uniform(2.0, 5.0)
            csi_distortion = np.random.uniform(2.0, 4.0)

        elif env_name == 'rain':
            # 降雨扰动
            rain_rate = np.random.uniform(10, 100)
            rain_loss = self.prop_model.rain_attenuation(freq_mhz, rain_rate, distance_m)
            extra_loss = rain_loss + np.random.normal(0, 2)
            snr_degradation = rain_loss * 0.5 + np.random.normal(0, 1)
            delay_spread_factor = np.random.uniform(1.5, 4.0)

            # 毫米波频段对降雨更敏感
            if 'mmwave' in band_name:
                extra_loss *= 2.5
                snr_degradation *= 2.0

        # 阴影衰落
        shadowing = self.prop_model.shadowing_fading()

        # 总路径损耗
        total_pl = pl_db + extra_loss + shadowing

        # RSS计算
        rss = tx_power_dbm + antenna_gain_db - total_pl

        # CSI生成
        n_taps = 16
        csi_real, csi_imag = self.prop_model.multipath_fading(n_taps)

        # 根据环境状态扭曲CSI
        if env_name != 'normal':
            for i in range(n_taps):
                csi_real[i] += np.random.normal(0, 0.3 * csi_distortion)
                csi_imag[i] += np.random.normal(0, 0.3 * csi_distortion)

        # 时延扩展
        delay_spread = 50 * delay_spread_factor * np.random.uniform(0.8, 1.2)

        # 多普勒频移
        base_doppler = 50  # Hz for drone movement
        doppler = base_doppler * doppler_factor * np.random.uniform(0.8, 1.2)

        # SNR
        snr = 20 + 2 * antenna_gain_db - total_pl * 0.5 - snr_degradation + np.random.normal(0, 2)

        # RSRP
        rsrp = rss + np.random.normal(0, 1)

        sample = {
            'features': {
                'rss': float(rss),
                'csi_real': [float(v) for v in csi_real],
                'csi_imag': [float(v) for v in csi_imag],
                'path_loss': float(total_pl),
                'delay_spread': float(delay_spread),
                'doppler': float(doppler),
                'snr': float(snr),
                'rsrp': float(rsrp),
            },
            'metadata': {
                'tx_pos': tx_pos,
                'rx_pos': rx_pos,
                'distance_m': float(distance_m),
                'frequency_mhz': freq_mhz,
                'band_name': band_name,
                'env_state_id': env_state_id,
                'env_state_name': env_name,
                'env_state_desc': env_config['description'],
            }
        }
        return sample

    def generate_dataset(self,
                         n_samples_per_env=2000,
                         n_base_stations=4,
                         area_size=1000,
                         output_dir='./data') -> dict:
        """
        生成完整的仿真数据集

        返回: dict with keys: 'train', 'test', 'metadata'
        """
        print(f"[DataGen] 开始生成仿真数据集...")
        print(f"  - 每环境状态样本数: {n_samples_per_env}")
        print(f"  - 基站数量: {n_base_stations}")
        print(f"  - 区域大小: {area_size}x{area_size} m^2")

        # 生成基站布局
        bs_positions = self.generate_base_station_layout(n_base_stations, area_size)
        print(f"  - 基站位置: {bs_positions}")

        all_samples = []

        for env_id, env_config in self.ENV_STATES.items():
            print(f"[DataGen] 生成环境: {env_config['name']} ({env_config['description']})")

            for i in range(n_samples_per_env):
                # 随机生成接收端（无人机）位置
                rx_x = np.random.uniform(0, area_size)
                rx_y = np.random.uniform(0, area_size)
                rx_z = np.random.uniform(10, 300)  # 飞行高度
                rx_pos = (rx_x, rx_y, rx_z)

                # 选择一个基站和频段
                bs_idx = np.random.randint(0, len(bs_positions))
                band_key = np.random.choice(list(self.BAND_CONFIG.keys()))

                tx_pos = bs_positions[bs_idx]

                sample = self.generate_single_sample(
                    tx_pos, rx_pos, band_key, env_id
                )
                sample['metadata']['bs_index'] = bs_idx
                sample['metadata']['bs_position'] = tx_pos
                all_samples.append(sample)

        # 划分训练/测试集
        np.random.shuffle(all_samples)
        split_idx = int(0.8 * len(all_samples))
        train_samples = all_samples[:split_idx]
        test_samples = all_samples[split_idx:]

        dataset = {
            'train': train_samples,
            'test': test_samples,
            'metadata': {
                'base_stations': bs_positions,
                'band_config': self.BAND_CONFIG,
                'env_states': self.ENV_STATES,
                'area_size': area_size,
                'n_samples_total': len(all_samples),
                'n_train': len(train_samples),
                'n_test': len(test_samples),
            }
        }

        # 保存
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'synthetic_dataset.pkl'), 'wb') as f:
            pickle.dump(dataset, f)

        # 同时保存JSON格式的元数据
        with open(os.path.join(output_dir, 'dataset_metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(dataset['metadata'], f, ensure_ascii=False, indent=2)

        print(f"[DataGen] 数据集生成完成!")
        print(f"  - 总样本数: {len(all_samples)}")
        print(f"  - 训练集: {len(train_samples)}, 测试集: {len(test_samples)}")
        print(f"  - 保存路径: {output_dir}")

        return dataset


if __name__ == '__main__':
    generator = SyntheticDataGenerator(random_seed=42)
    dataset = generator.generate_dataset(
        n_samples_per_env=1000,
        n_base_stations=4,
        area_size=1000,
        output_dir='./data'
    )
    print("\n示例样本:")
    sample = dataset['train'][0]
    print(f"  特征: RSS={sample['features']['rss']:.2f}dBm, "
          f"PathLoss={sample['features']['path_loss']:.2f}dB, "
          f"SNR={sample['features']['snr']:.2f}dB")
    print(f"  环境: {sample['metadata']['env_state_name']}")
    print(f"  频段: {sample['metadata']['band_name']} ({sample['metadata']['frequency_mhz']}MHz)")
