"""
无线环境态势地图生成模块
基于空间插值算法（Kriging、IDW、RBF等）和时序预测，
将离散观测点融合为连续的环境态势地图
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import Rbf, griddata
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as GPR_RBF, WhiteKernel
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class SpatialInterpolator:
    """空间插值器基类"""

    @staticmethod
    def idw_interpolation(points: np.ndarray, values: np.ndarray,
                          grid_resolution: Tuple[int, int],
                          area_size: Tuple[float, float],
                          power: float = 2.0) -> np.ndarray:
        """
        反距离加权插值 (IDW)

        points: (n_points, 2) 观测点坐标
        values: (n_points,) 观测值
        grid_resolution: (nx, ny) 网格分辨率
        area_size: (width, height) 区域大小（米）
        power: 距离幂次
        """
        nx, ny = grid_resolution
        width, height = area_size

        # 生成网格
        x_grid = np.linspace(0, width, nx)
        y_grid = np.linspace(0, height, ny)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # 计算距离
        distances = cdist(grid_points, points)

        # 避免除零
        distances = np.maximum(distances, 1e-10)

        # IDW权重
        weights = 1.0 / (distances ** power)
        weights_sum = weights.sum(axis=1, keepdims=True)

        # 加权平均
        interpolated = (weights @ values) / weights_sum.ravel()
        interpolated = interpolated.reshape(ny, nx)

        return interpolated

    @staticmethod
    def rbf_interpolation(points: np.ndarray, values: np.ndarray,
                          grid_resolution: Tuple[int, int],
                          area_size: Tuple[float, float],
                          function: str = 'multiquadric') -> np.ndarray:
        """
        径向基函数插值 (RBF)
        """
        nx, ny = grid_resolution
        width, height = area_size

        x_grid = np.linspace(0, width, nx)
        y_grid = np.linspace(0, height, ny)
        xx, yy = np.meshgrid(x_grid, y_grid)

        try:
            rbf = Rbf(points[:, 0], points[:, 1], values, function=function)
            interpolated = rbf(xx, yy)
        except Exception:
            # 如果RBF失败，回退到griddata
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            interpolated = griddata(points, values, grid_points, method='cubic')
            if interpolated is None or np.any(np.isnan(interpolated)):
                interpolated = griddata(points, values, grid_points, method='linear')
            interpolated = interpolated.reshape(ny, nx)

        return interpolated

    @staticmethod
    def kriging_interpolation(points: np.ndarray, values: np.ndarray,
                              grid_resolution: Tuple[int, int],
                              area_size: Tuple[float, float]) -> np.ndarray:
        """
        Kriging/GP插值
        """
        nx, ny = grid_resolution
        width, height = area_size

        # 使用高斯过程回归作为Kriging近似
        kernel = GPR_RBF(length_scale=50.0, length_scale_bounds=(10.0, 200.0)) + \
                 WhiteKernel(noise_level=0.1)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)

        gp.fit(points, values)

        x_grid = np.linspace(0, width, nx)
        y_grid = np.linspace(0, height, ny)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        interpolated, std = gp.predict(grid_points, return_std=True)
        interpolated = interpolated.reshape(ny, nx)

        return interpolated


class TemporalPredictor(nn.Module):
    """时序预测器 (LSTM)"""
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2, output_dim=4, seq_len=10):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 取最后一个时间步
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output


class RadioEnvironmentMap:
    """
    无线环境地图生成器
    综合空间插值和时序预测，生成连续区域态势图
    """

    ENV_STATE_COLORS = {
        0: 'green',   # normal
        1: 'orange',  # obstacle
        2: 'red',     # interference
        3: 'blue',    # rain
    }

    ENV_STATE_NAMES = {
        0: '正常',
        1: '障碍遮挡',
        2: '同频干扰',
        3: '降雨扰动',
    }

    def __init__(self,
                 area_size: Tuple[float, float] = (1000, 1000),
                 grid_resolution: Tuple[int, int] = (50, 50),
                 interpolator: str = 'rbf'):
        """
        area_size: (width, height) 区域尺寸（米）
        grid_resolution: (nx, ny) 网格分辨率
        interpolator: 'idw', 'rbf', 'kriging'
        """
        self.area_size = area_size
        self.grid_resolution = grid_resolution
        self.interpolator = interpolator
        self.spatial_interp = SpatialInterpolator()

        # 地图历史（用于时序预测）
        self.map_history = []
        self.max_history = 20

    def generate_map(self,
                     observation_points: List[Tuple[float, float]],
                     observation_values: np.ndarray,
                     interpolation_method: str = None) -> np.ndarray:
        """
        生成态势地图

        observation_points: List of (x, y) 观测点坐标
        observation_values: (n_points,) or (n_points, n_features) 观测值
        """
        points = np.array(observation_points)

        # 处理多维观测值
        if observation_values.ndim == 1:
            observation_values = observation_values.reshape(-1, 1)

        n_features = observation_values.shape[1]
        nx, ny = self.grid_resolution
        maps = []

        method = interpolation_method or self.interpolator

        for f in range(n_features):
            values = observation_values[:, f]

            if method == 'idw':
                interp_map = self.spatial_interp.idw_interpolation(
                    points, values, self.grid_resolution, self.area_size
                )
            elif method == 'kriging':
                interp_map = self.spatial_interp.kriging_interpolation(
                    points, values, self.grid_resolution, self.area_size
                )
            else:  # rbf (默认)
                interp_map = self.spatial_interp.rbf_interpolation(
                    points, values, self.grid_resolution, self.area_size
                )

            maps.append(interp_map)

        # 堆叠为 (ny, nx, n_features)
        stacked_map = np.stack(maps, axis=-1)

        # 保存历史
        self.map_history.append(stacked_map)
        if len(self.map_history) > self.max_history:
            self.map_history.pop(0)

        return stacked_map

    def generate_multi_station_fusion_map(self,
                                          station_observations: List[Dict]) -> np.ndarray:
        """
        多基站观测融合的地图生成

        station_observations: List of dict with keys:
            - 'position': (x, y, z) 基站位置
            - 'coverage_points': List of (x, y) 覆盖区域采样点
            - 'predicted_states': List of int 预测环境状态
            - 'confidences': List of float 置信度
        """
        all_points = []
        all_values = []
        all_weights = []

        for station in station_observations:
            cover_points = station['coverage_points']
            states = station['predicted_states']
            confidences = station.get('confidences', [1.0] * len(states))

            for pt, state, conf in zip(cover_points, states, confidences):
                all_points.append(pt)
                all_values.append(state)
                all_weights.append(conf)

        points = np.array(all_points)
        values = np.array(all_values, dtype=float)
        weights = np.array(all_weights)

        # 加权空间插值
        nx, ny = self.grid_resolution
        width, height = self.area_size

        x_grid = np.linspace(0, width, nx)
        y_grid = np.linspace(0, height, ny)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        distances = cdist(grid_points, points)
        distances = np.maximum(distances, 1e-10)

        weights_matrix = weights.reshape(1, -1)
        weights_tiled = np.repeat(weights_matrix, distances.shape[0], axis=0)

        w = weights_tiled / (distances ** 2)
        w_sum = w.sum(axis=1)

        interpolated = (w @ values) / w_sum
        interpolated = interpolated.reshape(ny, nx)

        # 保存历史
        single_channel = interpolated.reshape(ny, nx, 1)
        self.map_history.append(single_channel)
        if len(self.map_history) > self.max_history:
            self.map_history.pop(0)

        return interpolated

    def predict_future_map(self, n_steps: int = 1) -> List[np.ndarray]:
        """
        基于时序历史预测未来地图变化

        使用简化的趋势外推方法
        """
        if len(self.map_history) < 3:
            return [self.map_history[-1].copy()] * n_steps if self.map_history else []

        future_maps = []

        # 使用最近3帧进行线性趋势外推
        recent = self.map_history[-3:]

        for step in range(1, n_steps + 1):
            # 简单的时间差分外推
            if len(recent) >= 2:
                trend = recent[-1] - recent[-2]
                predicted = recent[-1] + trend * step * 0.5
                # 限制范围
                predicted = np.clip(predicted, 0, 3)
                future_maps.append(predicted)
            else:
                future_maps.append(recent[-1].copy())

        return future_maps

    def compute_coverage_rate(self, station_positions: List[Tuple[float, float]],
                              coverage_radius: float = 200.0) -> float:
        """
        计算有效区域覆盖率
        """
        nx, ny = self.grid_resolution
        width, height = self.area_size

        # 生成网格
        x = np.linspace(0, width, nx)
        y = np.linspace(0, height, ny)
        xx, yy = np.meshgrid(x, y)
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # 检查每个网格点是否被至少一个基站覆盖
        covered = np.zeros(len(grid_points), dtype=bool)
        for sp in station_positions:
            sp_2d = np.array([[sp[0], sp[1]]])
            dists = cdist(grid_points, sp_2d).ravel()
            covered |= (dists <= coverage_radius)

        coverage_rate = covered.sum() / len(grid_points)
        return coverage_rate

    def compute_reconstruction_error(self,
                                     true_map: np.ndarray,
                                     estimated_map: np.ndarray) -> dict:
        """
        计算地图重构误差
        """
        diff = true_map - estimated_map
        mae = np.mean(np.abs(diff))
        mse = np.mean(diff ** 2)
        rmse = np.sqrt(mse)

        # 将网格误差转换为空间误差（米）
        pixel_size_x = self.area_size[0] / self.grid_resolution[0]
        pixel_size_y = self.area_size[1] / self.grid_resolution[1]
        avg_pixel_size = (pixel_size_x + pixel_size_y) / 2

        spatial_error_m = rmse * avg_pixel_size

        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'spatial_error_m': spatial_error_m,
        }


if __name__ == '__main__':
    # 测试地图生成
    rem = RadioEnvironmentMap(area_size=(1000, 1000), grid_resolution=(50, 50))

    # 模拟观测点
    np.random.seed(42)
    n_points = 50
    points = [(np.random.uniform(0, 1000), np.random.uniform(0, 1000)) for _ in range(n_points)]
    values = np.random.choice([0, 1, 2, 3], size=n_points, p=[0.5, 0.2, 0.2, 0.1])

    map_data = rem.generate_map(points, values)
    print(f"地图形状: {map_data.shape}")
    print(f"地图范围: [{map_data.min():.2f}, {map_data.max():.2f}]")

    # 测试覆盖率
    station_pos = [(200, 200, 30), (800, 200, 30), (200, 800, 35), (800, 800, 35)]
    coverage = rem.compute_coverage_rate(station_pos, coverage_radius=300)
    print(f"单站覆盖率: {coverage*100:.1f}%")
