"""
Classification Memory Bank for SIREN framework.

Author: Yucong Zhang
Email: yucong0428@outlook.com

This module provides memory bank functionality for classification tasks,
storing feature-label pairs and supporting KNN query operations.
"""

import numpy as np
import torch
from .base_memory_bank import BaseMemoryBank
import os

class ClassificationMemoryBank(BaseMemoryBank):
    """
    Memory bank for classification tasks: stores feature-label pairs and supports KNN query.
    """
    def __init__(self):
        self.features = []  # List of feature vectors (numpy or torch)
        self.labels = []    # List of labels (int or str)

    def add(self, feature, label=None):
        self.features.append(np.array(feature))
        self.labels.append(label)

    def batch_add(self, features, labels=None):
        features = [np.array(f) for f in features]
        self.features.extend(features)
        if labels is not None:
            self.labels.extend(labels)
        else:
            self.labels.extend([None] * len(features))

    def clear(self):
        self.features = []
        self.labels = []

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, features=np.stack(self.features), labels=np.array(self.labels))

    def load(self, path):
        data = np.load(path, allow_pickle=True)
        self.features = [f for f in data['features']]
        self.labels = list(data['labels'])

    def knn_query(self, feature, k=1, metric='euclidean', fusion: str = 'mean', return_per_channel: bool = False,
                  channel_normalize: bool = True):
        """
        对输入特征做KNN，返回最近邻的标签和距离。
        
        支持统一的多通道距离计算：单通道作为多通道的特殊情况。
        
        Args:
            feature: 查询特征，支持格式:
                    - [feature_dim]: 旧格式，向后兼容
                    - [num_channels, feature_dim]: 新格式，统一多通道处理
            k: 返回最近邻数量
            metric: 距离度量方式 ('euclidean' 或 'cosine')
            fusion: 通道融合方式（当前支持 'mean' 或 'none'；主返回始终为mean以保持兼容）
            return_per_channel: 若为True，额外返回每通道的top-k结果
            
        Returns:
            - 默认: (nearest_labels, distances)
            - 若 return_per_channel=True: (nearest_labels, distances, per_channel_labels, per_channel_dists)
        """
        if len(self.features) == 0:
            raise ValueError("Memory bank is empty!")
        
        feature = np.array(feature)
        
        # 检测并标准化特征格式
        if feature.ndim == 1:
            # 旧格式 [feature_dim] -> 转换为 [1, feature_dim] 进行统一处理
            feature = feature.reshape(1, -1)
        elif feature.ndim == 2:
            # 新格式 [num_channels, feature_dim]，直接使用
            pass
        else:
            raise ValueError(f"Unsupported feature dimension: {feature.ndim}")
        
        # 同样标准化存储的特征
        feats = []
        for stored_feat in self.features:
            stored_feat = np.array(stored_feat)
            if stored_feat.ndim == 1:
                # 旧格式存储的特征
                stored_feat = stored_feat.reshape(1, -1)
            feats.append(stored_feat)
        feats = np.stack(feats)  # [num_samples, num_channels, feature_dim]
        
        # 统一的多通道距离计算 -> per-channel distances [N,C]
        num_channels = feature.shape[0]
        dists_per_channel = np.zeros((feats.shape[0], num_channels), dtype=np.float32)
        for ch in range(num_channels):
            query_ch = feature[ch]
            memory_ch = feats[:, ch]
            if metric == 'euclidean':
                d = np.linalg.norm(memory_ch - query_ch, axis=1)
            elif metric == 'cosine':
                memory_ch_norm = memory_ch / (np.linalg.norm(memory_ch, axis=1, keepdims=True) + 1e-8)
                query_ch_norm = query_ch / (np.linalg.norm(query_ch) + 1e-8)
                d = 1 - np.dot(memory_ch_norm, query_ch_norm)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            dists_per_channel[:, ch] = d
        
        # 通道融合前标准化（每通道对[N]做z-score），减少通道量纲差异的影响
        if channel_normalize and dists_per_channel.shape[1] > 1:
            mu = dists_per_channel.mean(axis=0, keepdims=True)
            sig = dists_per_channel.std(axis=0, keepdims=True) + 1e-8
            dists_for_fusion = (dists_per_channel - mu) / sig
        else:
            dists_for_fusion = dists_per_channel

        # 融合为主结果（保持兼容用mean）
        fused_distances = dists_for_fusion.mean(axis=1)
        
        # 返回最近邻
        idx = np.argsort(fused_distances)[:k]
        main_labels = [self.labels[i] for i in idx]
        main_dists = fused_distances[idx]
        
        if not return_per_channel:
            return main_labels, main_dists
        
        # 每通道top-k结果
        per_labels = []  # List[List[label]] length C each with k
        per_dists = []   # List[np.ndarray(k,)] per channel
        for ch in range(num_channels):
            idx_c = np.argsort(dists_per_channel[:, ch])[:k]
            per_labels.append([self.labels[i] for i in idx_c])
            per_dists.append(dists_per_channel[idx_c, ch])
        per_dists = [np.array(x) for x in per_dists]
        return main_labels, main_dists, per_labels, per_dists

    def knn_query_batch(self, features, k: int = 1, metric: str = 'euclidean', leave_one_out: bool = False, labels: list = None,
                        fusion: str = 'mean', return_per_channel: bool = False, channel_normalize: bool = True):
        """
        批量KNN查询，支持多通道融合与LOOCV自样本剔除。
        features: [B,C,D] 或 [B,D]
        返回: 
          - 默认 (topk_labels: List[List[label]], topk_distances: np.ndarray[B,k])
          - 若 return_per_channel=True，额外返回 (topk_labels_per_channel: List[B][C][k], topk_dists_per_channel: np.ndarray[B,C,k])
        """
        if len(self.features) == 0:
            raise ValueError("Memory bank is empty!")

        # 标准化库特征 -> [N,C,D]
        feats_mem = []
        for stored in self.features:
            arr = np.array(stored)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            feats_mem.append(arr)
        Y = np.stack(feats_mem, axis=0)

        # 标准化查询特征 -> [B,C,D]
        X = np.array(features)
        if X.ndim == 2:
            X = X[:, None, :]
        if X.ndim != 3:
            raise ValueError(f"Unsupported batch feature shape: {X.shape}")

        # 对齐通道数
        if X.shape[1] != Y.shape[1]:
            if X.shape[1] == 1:
                X = np.repeat(X, Y.shape[1], axis=1)
            elif Y.shape[1] == 1:
                Y = np.repeat(Y, X.shape[1], axis=1)
            else:
                raise ValueError(f"Channel mismatch: X={X.shape}, Y={Y.shape}")

        # 计算距离（保留每通道），随后可做通道标准化再平均 -> [B,N]
        if metric == 'euclidean':
            x2 = np.sum(X * X, axis=2)                 # [B,C]
            y2 = np.sum(Y * Y, axis=2)                 # [N,C]
            xy = np.einsum('bcd,ncd->bnc', X, Y)       # [B,N,C]
            dist2 = x2[:, None, :] + y2[None, :, :] - 2.0 * xy
            dist2 = np.maximum(dist2, 0.0)
            dist = np.sqrt(dist2)                      # [B,N,C]
            per = dist[:,:,:7]
        elif metric == 'cosine':
            Xn = X / (np.linalg.norm(X, axis=2, keepdims=True) + 1e-10)
            Yn = Y / (np.linalg.norm(Y, axis=2, keepdims=True) + 1e-10)
            sim = np.einsum('bcd,ncd->bnc', Xn, Yn)    # [B,N,C]
            per = (1.0 - sim[:,:,:7])
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        # 通道标准化（z-score per (B,C) over N）
        if channel_normalize and (X.shape[1] > 1):
            mu = per.mean(axis=1, keepdims=True)       # [B,1,C]
            sig = per.std(axis=1, keepdims=True) + 1e-10
            per_norm = (per - mu) / sig
        else:
            per_norm = per

        fused = per_norm.mean(axis=2)                  # [B,N]

        # LOOCV自样本剔除: 若B==N，掩码对角线
        if leave_one_out and fused.shape[0] == fused.shape[1]:
            print("Warning: Fill diagonal with np.inf to avoid self-comparison")
            np.fill_diagonal(fused, np.inf)

        # 取top-k
        k = min(k, fused.shape[1])
        topk_idx = np.argpartition(fused, kth=k-1, axis=1)[:, :k]
        rows = np.arange(fused.shape[0])[:, None]
        topk_sorted_idx = topk_idx[rows, np.argsort(fused[rows, topk_idx], axis=1)]
        topk_dists = fused[rows, topk_sorted_idx]
        lbls = labels if labels is not None else self.labels
        topk_labels = [[lbls[j] for j in row] for row in topk_sorted_idx]
        if not return_per_channel:
            return topk_labels, topk_dists
        
        # 每通道top-k（向量化按通道选择） -> [B,C,k]
        B, N = fused.shape
        C = X.shape[1]
        topk_idx_ch = np.empty((B, C, k), dtype=int)
        topk_dists_ch = np.empty((B, C, k), dtype=float)
        # 逐通道选择top-k，C<=16，循环可接受
        per_raw = dist if metric == 'euclidean' else (1.0 - sim)  # [B,N,C]
        for c in range(C):
            per_c = per_raw[:, :, c]  # [B,N]
            idx_part = np.argpartition(per_c, kth=k-1, axis=1)[:, :k]
            # sort these k per row
            rows = np.arange(B)[:, None]
            sorted_idx = idx_part[rows, np.argsort(per_c[rows, idx_part], axis=1)]
            topk_idx_ch[:, c, :] = sorted_idx
            topk_dists_ch[:, c, :] = per_c[rows, sorted_idx]
        topk_labels_ch = [[[lbls[j] for j in topk_idx_ch[b, c, :]] for c in range(C)] for b in range(B)]
        return topk_labels, topk_dists, topk_labels_ch, topk_dists_ch

    def knn_query_batch_chunked(self, features, k: int = 1, metric: str = 'euclidean', 
                               chunk_size: int = 100, leave_one_out: bool = False, 
                               labels: list = None, fusion: str = 'mean', 
                               return_per_channel: bool = False, channel_normalize: bool = True):
        """
        分块KNN查询，控制内存使用
        
        Args:
            features: 查询特征 [B,C,D] 或 [B,D]
            k: 返回最近邻数量
            metric: 距离度量方式
            chunk_size: 每次处理的测试样本数量，控制内存使用
            leave_one_out: 是否进行留一法交叉验证
            labels: 标签列表
            fusion: 通道融合方式
            return_per_channel: 是否返回每通道的top-k结果
            channel_normalize: 是否进行通道标准化
            
        Returns:
            与原始方法完全相同的返回格式
        """
        if len(self.features) == 0:
            raise ValueError("Memory bank is empty!")
        
        # 标准化查询特征
        X = np.array(features)
        if X.ndim == 2:
            X = X[:, None, :]
        if X.ndim != 3:
            raise ValueError(f"Unsupported batch feature shape: {X.shape}")
        
        total_samples = X.shape[0]
        
        # 如果样本数量小于chunk_size，直接使用原始方法
        if total_samples <= chunk_size:
            return self.knn_query_batch(features, k, metric, leave_one_out, 
                                      labels, fusion, return_per_channel, channel_normalize)
        
        # 分块处理
        all_topk_labels = []
        all_topk_distances = []
        all_topk_labels_per_channel = []
        all_topk_dists_per_channel = []
        
        for start_idx in range(0, total_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, total_samples)
            chunk_features = X[start_idx:end_idx]
            
            # 处理当前chunk
            chunk_results = self._knn_query_single_chunk(
                chunk_features, k, metric, leave_one_out, labels, 
                fusion, return_per_channel, channel_normalize
            )
            
            # 收集结果
            if return_per_channel:
                chunk_labels, chunk_dists, chunk_labels_ch, chunk_dists_ch = chunk_results
                all_topk_labels.extend(chunk_labels)
                all_topk_distances.extend(chunk_dists)
                all_topk_labels_per_channel.extend(chunk_labels_ch)
                all_topk_dists_per_channel.extend(chunk_dists_ch)
            else:
                chunk_labels, chunk_dists = chunk_results
                all_topk_labels.extend(chunk_labels)
                all_topk_distances.extend(chunk_dists)
        
        # 转换为numpy数组
        all_topk_distances = np.array(all_topk_distances)
        
        if return_per_channel:
            all_topk_dists_per_channel = np.array(all_topk_dists_per_channel)
            return all_topk_labels, all_topk_distances, all_topk_labels_per_channel, all_topk_dists_per_channel
        else:
            return all_topk_labels, all_topk_distances

    def _knn_query_single_chunk(self, chunk_features, k, metric, leave_one_out, 
                               labels, fusion, return_per_channel, channel_normalize):
        """
        处理单个chunk的KNN查询，保持与原始方法完全相同的逻辑
        
        Args:
            chunk_features: 当前chunk的特征 [chunk_size, C, D]
            k: 最近邻数量
            metric: 距离度量方式
            leave_one_out: 是否进行留一法交叉验证
            labels: 标签列表
            fusion: 通道融合方式
            return_per_channel: 是否返回每通道结果
            channel_normalize: 是否进行通道标准化
            
        Returns:
            与原始方法相同的返回格式
        """
        # 标准化库特征 -> [N,C,D]
        feats_mem = []
        for stored in self.features:
            arr = np.array(stored)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            feats_mem.append(arr)
        Y = np.stack(feats_mem, axis=0)
        
        # 对齐通道数
        if chunk_features.shape[1] != Y.shape[1]:
            if chunk_features.shape[1] == 1:
                chunk_features = np.repeat(chunk_features, Y.shape[1], axis=1)
            elif Y.shape[1] == 1:
                Y = np.repeat(Y, chunk_features.shape[1], axis=1)
            else:
                raise ValueError(f"Channel mismatch: chunk_features={chunk_features.shape}, Y={Y.shape}")
        
        # 计算距离（与原始方法完全相同的逻辑）
        if metric == 'euclidean':
            x2 = np.sum(chunk_features * chunk_features, axis=2)     # [chunk_size,C]
            y2 = np.sum(Y * Y, axis=2)                             # [N,C]
            xy = np.einsum('bcd,ncd->bnc', chunk_features, Y)      # [chunk_size,N,C]
            dist2 = x2[:, None, :] + y2[None, :, :] - 2.0 * xy
            dist2 = np.maximum(dist2, 0.0)
            dist = np.sqrt(dist2)                                  # [chunk_size,N,C]
            per = dist[:,:,:7]
        elif metric == 'cosine':
            Xn = chunk_features / (np.linalg.norm(chunk_features, axis=2, keepdims=True) + 1e-10)
            Yn = Y / (np.linalg.norm(Y, axis=2, keepdims=True) + 1e-10)
            sim = np.einsum('bcd,ncd->bnc', Xn, Yn)                # [chunk_size,N,C]
            per = (1.0 - sim[:,:,:7])
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        # 通道标准化（与原始方法完全相同）
        if channel_normalize and (chunk_features.shape[1] > 1):
            mu = per.mean(axis=1, keepdims=True)                   # [chunk_size,1,C]
            sig = per.std(axis=1, keepdims=True) + 1e-10
            per_norm = (per - mu) / sig
        else:
            per_norm = per
        
        fused = per_norm.mean(axis=2)                              # [chunk_size,N]
        
        # LOOCV自样本剔除
        if leave_one_out and fused.shape[0] == fused.shape[1]:
            np.fill_diagonal(fused, np.inf)
        
        # 取top-k（与原始方法完全相同）
        k = min(k, fused.shape[1])
        topk_idx = np.argpartition(fused, kth=k-1, axis=1)[:, :k]
        rows = np.arange(fused.shape[0])[:, None]
        topk_sorted_idx = topk_idx[rows, np.argsort(fused[rows, topk_idx], axis=1)]
        topk_dists = fused[rows, topk_sorted_idx]
        
        lbls = labels if labels is not None else self.labels
        topk_labels = [[lbls[j] for j in row] for row in topk_sorted_idx]
        
        if not return_per_channel:
            return topk_labels, topk_dists
        
        # 每通道top-k（与原始方法完全相同）
        chunk_size, N = fused.shape
        C = chunk_features.shape[1]
        topk_idx_ch = np.empty((chunk_size, C, k), dtype=int)
        topk_dists_ch = np.empty((chunk_size, C, k), dtype=float)
        
        per_raw = dist if metric == 'euclidean' else (1.0 - sim)
        for c in range(C):
            per_c = per_raw[:, :, c]
            idx_part = np.argpartition(per_c, kth=k-1, axis=1)[:, :k]
            rows = np.arange(chunk_size)[:, None]
            sorted_idx = idx_part[rows, np.argsort(per_c[rows, idx_part], axis=1)]
            topk_idx_ch[:, c, :] = sorted_idx
            topk_dists_ch[:, c, :] = per_c[rows, sorted_idx]
        
        topk_labels_ch = [[[lbls[j] for j in topk_idx_ch[b, c, :]] for c in range(C)] for b in range(chunk_size)]
        
        return topk_labels, topk_dists, topk_labels_ch, topk_dists_ch