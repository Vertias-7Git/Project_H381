"""
环境状态识别模型
基于深度神经网络，实现多频多基站无线特征到环境状态的映射
支持单站和联合决策两种模式
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
import pickle


class CSIEncoder(nn.Module):
    """CSI特征编码器 - 将复数CSI序列编码为固定维度向量"""
    def __init__(self, n_taps=16, output_dim=32):
        super().__init__()
        # 输入: (batch, n_taps*2) [real, imag]
        self.conv = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(32, output_dim)

    def forward(self, csi_real, csi_imag):
        # csi_real: (batch, n_taps)
        # csi_imag: (batch, n_taps)
        x = torch.stack([csi_real, csi_imag], dim=1)  # (batch, 2, n_taps)
        x = self.conv(x)  # (batch, 32, 1)
        x = x.squeeze(-1)  # (batch, 32)
        x = self.fc(x)     # (batch, output_dim)
        return x


class SingleStationClassifier(nn.Module):
    """单基站环境状态分类器"""
    def __init__(self,
                 n_taps=16,
                 csi_dim=32,
                 n_env_states=4,
                 dropout=0.3):
        super().__init__()

        self.csi_encoder = CSIEncoder(n_taps, csi_dim)

        # 频带嵌入（3种频段 -> embedding）
        self.band_embedding = nn.Embedding(3, 8)

        # 特征融合
        # 输入特征: RSS(1) + CSI_dim(32) + PL(1) + DS(1) + Doppler(1) + SNR(1) + RSRP(1) + Band_Embed(8)
        feature_dim = 1 + csi_dim + 1 + 1 + 1 + 1 + 1 + 8

        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(64, n_env_states)

    def forward(self, features, band_ids):
        """
        features: dict with keys: rss, csi_real, csi_imag, path_loss, delay_spread, doppler, snr, rsrp
        band_ids: (batch,) int tensor, 0=sub_6ghz, 1=mmwave_28, 2=mmwave_60
        """
        batch_size = features['rss'].shape[0]

        # CSI编码
        csi_encoded = self.csi_encoder(features['csi_real'], features['csi_imag'])

        # 频段嵌入
        band_embed = self.band_embedding(band_ids)

        # 拼接所有特征
        x = torch.cat([
            features['rss'].unsqueeze(1),
            csi_encoded,
            features['path_loss'].unsqueeze(1),
            features['delay_spread'].unsqueeze(1),
            features['doppler'].unsqueeze(1),
            features['snr'].unsqueeze(1),
            features['rsrp'].unsqueeze(1),
            band_embed,
        ], dim=1)

        x = self.feature_extractor(x)
        logits = self.classifier(x)
        return logits


class MultiStationFusion(nn.Module):
    """多基站联合感知模型 - 融合多个基站的观测结果"""
    def __init__(self,
                 single_station_model: SingleStationClassifier,
                 n_stations=4,
                 fusion_method='attention'):
        super().__init__()
        self.single_station = single_station_model
        self.n_stations = n_stations
        self.fusion_method = fusion_method

        # 基站间注意力融合
        self.station_attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            batch_first=True
        )

        # 位置编码（基站空间位置信息）
        self.position_embedding = nn.Linear(3, 16)

        # 联合决策层
        self.fusion_mlp = nn.Sequential(
            nn.Linear(64 + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.classifier = nn.Linear(64, 4)

    def forward(self, station_features_list, station_positions, station_band_ids_list):
        """
        station_features_list: List of dict, length = n_stations (可能包含None)
        station_positions: (n_stations, 3) spatial positions
        station_band_ids_list: List of band_id tensors
        """
        # 找到第一个非None的特征来获取batch_size和device
        ref_features = next((s for s in station_features_list if s is not None), None)
        if ref_features is None:
            raise ValueError("All station features are None")
        batch_size = ref_features['rss'].shape[0]
        device = ref_features['rss'].device

        # 提取各基站特征
        station_embeddings = []
        for features, band_ids in zip(station_features_list, station_band_ids_list):
            if features is None:
                # 缺失基站用零填充
                emb = torch.zeros(batch_size, 64, device=device)
            else:
                # 通过单站模型提取特征（到BN之后的那层）
                csi_encoded = self.single_station.csi_encoder(features['csi_real'], features['csi_imag'])
                band_embed = self.single_station.band_embedding(band_ids)
                x = torch.cat([
                    features['rss'].unsqueeze(1),
                    csi_encoded,
                    features['path_loss'].unsqueeze(1),
                    features['delay_spread'].unsqueeze(1),
                    features['doppler'].unsqueeze(1),
                    features['snr'].unsqueeze(1),
                    features['rsrp'].unsqueeze(1),
                    band_embed,
                ], dim=1)
                emb = self.single_station.feature_extractor(x)
            station_embeddings.append(emb)

        # 堆叠为序列
        station_embeddings = torch.stack(station_embeddings, dim=1)  # (batch, n_stations, 64)

        # 位置编码
        pos_embed = self.position_embedding(station_positions)  # (n_stations, 16)
        pos_embed = pos_embed.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, n_stations, 16)

        # 注意力融合
        attn_out, attn_weights = self.station_attention(
            station_embeddings, station_embeddings, station_embeddings
        )
        # attn_out: (batch, n_stations, 64)
        # attn_weights: (batch, n_stations, n_stations)

        # 加权聚合（可学习的）
        fused = attn_out.mean(dim=1)  # (batch, 64)

        # 结合位置信息
        pos_mean = pos_embed.mean(dim=1)  # (batch, 16)
        combined = torch.cat([fused, pos_mean], dim=1)

        x = self.fusion_mlp(combined)
        logits = self.classifier(x)

        return logits, attn_weights


class EnvironmentStateClassifier:
    """环境状态识别器 - 封装训练和推理逻辑"""

    BAND_NAME_TO_ID = {
        'sub_6ghz': 0,
        'mmwave_28': 1,
        'mmwave_60': 2,
    }

    ENV_STATE_NAMES = ['normal', 'obstacle', 'interference', 'rain']

    def __init__(self, model: nn.Module = None, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        if model is None:
            self.model = SingleStationClassifier().to(self.device)
        else:
            self.model = model.to(self.device)

    def prepare_features(self, sample: dict) -> Tuple[dict, int]:
        """将原始样本转换为模型输入格式"""
        features = sample['features']
        metadata = sample['metadata']

        # 转换为tensor
        feature_tensors = {
            'rss': torch.tensor([features['rss']], dtype=torch.float32),
            'csi_real': torch.tensor(features['csi_real'], dtype=torch.float32).unsqueeze(0),
            'csi_imag': torch.tensor(features['csi_imag'], dtype=torch.float32).unsqueeze(0),
            'path_loss': torch.tensor([features['path_loss']], dtype=torch.float32),
            'delay_spread': torch.tensor([features['delay_spread']], dtype=torch.float32),
            'doppler': torch.tensor([features['doppler']], dtype=torch.float32),
            'snr': torch.tensor([features['snr']], dtype=torch.float32),
            'rsrp': torch.tensor([features['rsrp']], dtype=torch.float32),
        }

        # 移动到设备
        feature_tensors = {k: v.to(self.device) for k, v in feature_tensors.items()}

        band_id = self.BAND_NAME_TO_ID[metadata['band_name']]
        band_tensor = torch.tensor([band_id], dtype=torch.long).to(self.device)

        return feature_tensors, band_tensor

    def predict(self, sample: dict, return_probs=False):
        """单样本预测"""
        self.model.eval()
        features, band_id = self.prepare_features(sample)

        with torch.no_grad():
            logits = self.model(features, band_id)
            probs = F.softmax(logits, dim=1)
            pred_id = torch.argmax(probs, dim=1).item()

        result = {
            'predicted_state': self.ENV_STATE_NAMES[pred_id],
            'predicted_id': pred_id,
            'confidence': probs[0, pred_id].item(),
        }

        if return_probs:
            result['probabilities'] = {
                name: probs[0, i].item()
                for i, name in enumerate(self.ENV_STATE_NAMES)
            }

        return result

    def predict_batch(self, samples: List[dict], return_probs=False):
        """批量预测"""
        self.model.eval()

        # 堆叠特征
        batch_features = {
            'rss': torch.stack([torch.tensor([s['features']['rss']]) for s in samples]).squeeze().to(self.device),
            'csi_real': torch.stack([torch.tensor(s['features']['csi_real']) for s in samples]).to(self.device),
            'csi_imag': torch.stack([torch.tensor(s['features']['csi_imag']) for s in samples]).to(self.device),
            'path_loss': torch.stack([torch.tensor([s['features']['path_loss']]) for s in samples]).squeeze().to(self.device),
            'delay_spread': torch.stack([torch.tensor([s['features']['delay_spread']]) for s in samples]).squeeze().to(self.device),
            'doppler': torch.stack([torch.tensor([s['features']['doppler']]) for s in samples]).squeeze().to(self.device),
            'snr': torch.stack([torch.tensor([s['features']['snr']]) for s in samples]).squeeze().to(self.device),
            'rsrp': torch.stack([torch.tensor([s['features']['rsrp']]) for s in samples]).squeeze().to(self.device),
        }

        # 确保二维
        for key in ['rss', 'path_loss', 'delay_spread', 'doppler', 'snr', 'rsrp']:
            if batch_features[key].dim() == 0:
                batch_features[key] = batch_features[key].unsqueeze(0)
            elif batch_features[key].dim() == 1:
                pass  # already (batch,)

        band_ids = torch.tensor([
            self.BAND_NAME_TO_ID[s['metadata']['band_name']]
            for s in samples
        ], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.model(batch_features, band_ids)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

        results = []
        for i in range(len(samples)):
            result = {
                'predicted_state': self.ENV_STATE_NAMES[preds[i].item()],
                'predicted_id': preds[i].item(),
                'confidence': probs[i, preds[i]].item(),
                'true_state': samples[i]['metadata']['env_state_name'],
                'true_id': samples[i]['metadata']['env_state_id'],
            }
            if return_probs:
                result['probabilities'] = {
                    name: probs[i, j].item()
                    for j, name in enumerate(self.ENV_STATE_NAMES)
                }
            results.append(result)

        return results

    def train_model(self,
                    train_samples: List[dict],
                    test_samples: List[dict],
                    epochs=50,
                    batch_size=64,
                    lr=1e-3,
                    model_save_path='./models/env_classifier.pth'):
        """训练模型"""
        print(f"[Model] 开始训练环境状态识别模型")
        print(f"[Model] 训练样本: {len(train_samples)}, 测试样本: {len(test_samples)}")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0
        history = {'train_loss': [], 'val_acc': [], 'val_f1': []}

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            n_batches = 0

            # Shuffle
            indices = np.random.permutation(len(train_samples))
            for i in range(0, len(train_samples), batch_size):
                batch_idx = indices[i:min(i+batch_size, len(train_samples))]
                batch_samples = [train_samples[j] for j in batch_idx]

                # 准备batch
                batch_features = {
                    'rss': torch.stack([
                        torch.tensor([s['features']['rss']]) for s in batch_samples
                    ]).squeeze().to(self.device),
                    'csi_real': torch.stack([
                        torch.tensor(s['features']['csi_real']) for s in batch_samples
                    ]).to(self.device),
                    'csi_imag': torch.stack([
                        torch.tensor(s['features']['csi_imag']) for s in batch_samples
                    ]).to(self.device),
                    'path_loss': torch.stack([
                        torch.tensor([s['features']['path_loss']]) for s in batch_samples
                    ]).squeeze().to(self.device),
                    'delay_spread': torch.stack([
                        torch.tensor([s['features']['delay_spread']]) for s in batch_samples
                    ]).squeeze().to(self.device),
                    'doppler': torch.stack([
                        torch.tensor([s['features']['doppler']]) for s in batch_samples
                    ]).squeeze().to(self.device),
                    'snr': torch.stack([
                        torch.tensor([s['features']['snr']]) for s in batch_samples
                    ]).squeeze().to(self.device),
                    'rsrp': torch.stack([
                        torch.tensor([s['features']['rsrp']]) for s in batch_samples
                    ]).squeeze().to(self.device),
                }

                for key in ['rss', 'path_loss', 'delay_spread', 'doppler', 'snr', 'rsrp']:
                    if batch_features[key].dim() == 0:
                        batch_features[key] = batch_features[key].unsqueeze(0)

                band_ids = torch.tensor([
                    self.BAND_NAME_TO_ID[s['metadata']['band_name']]
                    for s in batch_samples
                ], dtype=torch.long).to(self.device)

                labels = torch.tensor([
                    s['metadata']['env_state_id'] for s in batch_samples
                ], dtype=torch.long).to(self.device)

                # 前向传播
                logits = self.model(batch_features, band_ids)
                loss = criterion(logits, labels)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / n_batches

            # 验证
            val_results = self.predict_batch(test_samples)
            val_acc = np.mean([r['predicted_id'] == r['true_id'] for r in val_results])

            # F1 score
            from sklearn.metrics import f1_score, precision_score, recall_score
            y_true = [r['true_id'] for r in val_results]
            y_pred = [r['predicted_id'] for r in val_results]
            val_f1 = f1_score(y_true, y_pred, average='weighted')

            history['train_loss'].append(avg_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)

            scheduler.step(val_acc)

            print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}, "
                  f"Val Acc: {val_acc*100:.2f}%, Val F1: {val_f1:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), model_save_path)
                print(f"  -> 保存最佳模型 (Acc={best_acc*100:.2f}%)")

        print(f"[Model] 训练完成! 最佳验证准确率: {best_acc*100:.2f}%")
        return history


if __name__ == '__main__':
    # 测试模型结构
    model = SingleStationClassifier()
    print("单站分类器结构:")
    print(model)
    print(f"参数数量: {sum(p.numel() for p in model.parameters())}")
