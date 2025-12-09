import datetime
import time
from sklearn import metrics
import torch
import numpy as np
from typing import Sequence
import pandas as pd
from typing import Tuple, Sequence
from sklearn.feature_selection import mutual_info_regression
from sklearn.cluster import AgglomerativeClustering
import src.utils.log as log

_GLOBAL_COL_ORDER = None  

def flatten(x: torch.Tensor) -> torch.Tensor:
    """x: [N,D] 或 [N,T,D] -> [N*,D]"""
    if x.dim() == 2:
        return x
    if x.dim() == 3:
        n, t, d = x.shape
        return x.reshape(n * t, d)
    raise ValueError("只支持 [N,D] 或 [N,T,D]")


def mi_matrix(x: torch.Tensor, seed: int = 0) -> np.ndarray:
    """互信息矩阵 mi: [D,D]（包含 target）"""
    x2 = flatten(x).detach().cpu().numpy()  # [N*,D]
    _, d = x2.shape
    mi = np.zeros((d, d), dtype=np.float32)

    for i in range(d):
        for j in range(i + 1, d):
            v = mutual_info_regression(
                x2[:, [i]], x2[:, j], random_state=seed
            )[0]
            mi[i, j] = v
            mi[j, i] = v

    m = mi.max()
    if m > 0:
        mi = mi / m
    np.fill_diagonal(mi, 1.0)
    return mi


def periodic_score(x: torch.Tensor, lags: Sequence[int]) -> np.ndarray:
    """自相关做周期性评分，返回 scores: [D]"""
    x2 = flatten(x).detach().cpu().numpy()  # [N*,D]
    n, d = x2.shape

    x2 = x2 - x2.mean(axis=0, keepdims=True)
    var = np.var(x2, axis=0) + 1e-8

    scores = np.zeros(d, dtype=np.float32)
    for lag in lags:
        if lag <= 0 or lag >= n:
            continue
        num = (x2[lag:] * x2[:-lag]).sum(axis=0)  # [D]
        den = (n - lag) * var
        ac = num / den
        scores = np.maximum(scores, ac.astype(np.float32))

    return np.clip(scores, 0.0, 1.0)


def build_perm_periodic(
    x: torch.Tensor,
    n_clusters: int,
    lags: Sequence[int],
    linkage: str = "average",
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """
    x 不包含时间列，只包含数值特征和最后一列 target。
    x[..., -1] 是 target，不重排，只固定在最后。
    """
    d = x.shape[-1]
    if d < 2:
        raise ValueError("需要至少 2 列：特征 + target")

    mi = mi_matrix(x, seed)          # [D,D]
    per = periodic_score(x, lags)    # [D]

    dist = 1.0 - mi
    clu = AgglomerativeClustering(
       n_clusters=n_clusters,
       metric="precomputed",
       linkage=linkage,
    )
    labels_np = clu.fit_predict(dist)  # [D]

    # 中间特征 0..D-2，最后一列为 target
    mid_idx = np.arange(0, d - 1)
    mid_labels = labels_np[mid_idx]

    clusters = []
    for cid in range(n_clusters):
        idx = mid_idx[mid_labels == cid]
        if idx.size == 0:
            continue
        p_mean = per[idx].mean()
        clusters.append({"idx": idx, "p_mean": p_mean})

    if not clusters:
        perm_mid = mid_idx.copy()
    else:
        clusters.sort(key=lambda c: c["p_mean"], reverse=True)

        parts = []
        for info in clusters:
            idx = info["idx"]
            s = per[idx]
            order = np.argsort(-s)   # 周期性强的在左
            parts.append(idx[order])

        perm_mid = np.concatenate(parts, axis=0).astype(np.int64)

    perm = torch.empty(d, dtype=torch.long)
    perm[:-1] = torch.from_numpy(perm_mid)
    perm[-1] = d - 1                 # target 固定在最后

    labels = torch.from_numpy(labels_np.astype(np.int64))
    return perm, labels, mi, per


def reorder_features(
    df: pd.DataFrame,
    n_clusters: int = 4,
    lags: Sequence[int] = (24,),
    linkage: str = "average",
    seed: int = 0,
) -> pd.DataFrame:
    """
    df:
      - 第 0 列：时间（不参与计算，不重排）
      - 最后一列：target（数值，参与 MI/周期性，但固定在最后）
      - 中间列：数值特征（参与计算并重排）

    只在第一次调用时计算列顺序并缓存，
    后续调用直接按缓存列顺序重排。
    """
    global _GLOBAL_COL_ORDER

    cols = list(df.columns)
    if len(cols) < 2:
        return df

    # 如果已经算过一次顺序，后面直接用缓存
    if _GLOBAL_COL_ORDER is not None:
        # 假设所有 split 的列名一致
        return df[_GLOBAL_COL_ORDER].copy()

    # ====== 第一次调用：真正计算顺序 ======

    start_time = time.time()

    time_col = cols[0]
    target_col = cols[-1]
    feat_cols = cols[1:-1]    # 中间特征列

    if len(feat_cols) == 0:
        _GLOBAL_COL_ORDER = cols
        return df

    # 数值特征 + target -> tensor
    val_cols = feat_cols + [target_col]         # 这些列对应 x 的最后一维
    x_np = df[val_cols].to_numpy(dtype=float)   # [N, D_val]
    x = torch.from_numpy(x_np).float()

    perm, _, _, _ = build_perm_periodic(
        x,
        n_clusters=n_clusters,
        lags=lags,
        linkage=linkage,
        seed=seed,
    )  # perm: [D_val]

    end_time = time.time()

    log.Logger.log({
        'epoch': 'reorder',
        'cost_time': end_time - start_time,
        })

    print(f"重新排后的顺序:{perm}共耗时{end_time - start_time}s")

    perm_np = perm.cpu().numpy()
    mid_perm = perm_np[:-1]                      # 中间特征的新顺序（在 val_cols 的索引）
    feat_cols_new = [val_cols[i] for i in mid_perm]

    # 最终列顺序：时间 + 新特征顺序 + target
    new_cols = [time_col] + feat_cols_new + [target_col]

    # 缓存起来，后面 val/test 直接用
    _GLOBAL_COL_ORDER = new_cols

    return df[new_cols].copy()