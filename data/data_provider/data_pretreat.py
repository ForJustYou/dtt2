import datetime
import time
from sklearn import metrics
import torch
import numpy as np
from typing import Sequence, Tuple
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.cluster import AgglomerativeClustering
import src.utils.log as log

# 全局列顺序缓存
_GLOBAL_COL_ORDER = None

# ===== MI 过滤相关全局变量 =====
# 是否启用“根据与 target 的互信息做特征过滤（伪删除）”
_GLOBAL_MI_FILTER_ENABLED = True

# 与 target 的最小互信息阈值，小于该值的特征将被视为“低信息特征”
_MI_FILTER_THRESHOLD = 0.05

# 被视为“低信息”的特征列名列表（中间特征列，已经按列名记录）
_GLOBAL_MI_DROPPED_COLS = None

# 伪删除模式：
#   - "zero"   : 填 0
#   - "mean"   : 填该列均值（推荐）
#   - "shuffle": 打乱该列（只在第一次调用时有效，后续调用会每次重新打乱）
_MI_FILTER_MODE = "mean"

# 对于 "mean" 模式，第一次调用时计算并缓存各列的填充值，
# 后续 val/test 使用同一个填充值，保持一致性。
_GLOBAL_MI_FILL_VALUES = {}


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
        # 先按簇的周期性均值排序，周期性强的簇靠前
        clusters.sort(key=lambda c: c["p_mean"], reverse=True)

        parts = []
        for info in clusters:
            idx = info["idx"]
            s = per[idx]
            # 簇内再按单个特征的周期性从大到小排列
            order = np.argsort(-s)
            parts.append(idx[order])

        perm_mid = np.concatenate(parts, axis=0).astype(np.int64)

    perm = torch.empty(d, dtype=torch.long)
    perm[:-1] = torch.from_numpy(perm_mid)
    perm[-1] = d - 1                 # target 固定在最后

    labels = torch.from_numpy(labels_np.astype(np.int64))
    return perm, labels, mi, per


def _apply_pseudo_delete(df_new: pd.DataFrame) -> None:
    """
    对 df_new 就地应用“伪删除”：
    - 列数不变
    - 对 _GLOBAL_MI_DROPPED_COLS 中的列，根据 _MI_FILTER_MODE 做处理
    """
    global _GLOBAL_MI_DROPPED_COLS, _GLOBAL_MI_FILL_VALUES, _MI_FILTER_MODE

    if not _GLOBAL_MI_DROPPED_COLS:
        return

    for c in _GLOBAL_MI_DROPPED_COLS:
        if c not in df_new.columns:
            continue

        if _MI_FILTER_MODE == "zero":
            df_new[c] = 0.0

        elif _MI_FILTER_MODE == "mean":
            # 训练集第一次调用时会把均值存到 _GLOBAL_MI_FILL_VALUES
            if c in _GLOBAL_MI_FILL_VALUES:
                fill_val = _GLOBAL_MI_FILL_VALUES[c]
            else:
                fill_val = float(df_new[c].mean())
                _GLOBAL_MI_FILL_VALUES[c] = fill_val
            df_new[c] = fill_val

        elif _MI_FILTER_MODE == "shuffle":
            # 每次调用都打乱当前 df 中这一列的行顺序
            values = df_new[c].values
            perm_idx = np.random.permutation(len(values))
            df_new[c] = values[perm_idx]

        else:
            # 未知模式时，默认什么都不做
            pass


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
    后续调用直接按缓存列顺序重排，并对低 MI 特征做同样的伪删除处理。
    """
    global _GLOBAL_COL_ORDER, _GLOBAL_MI_DROPPED_COLS
    global _GLOBAL_MI_FILTER_ENABLED

    cols = list(df.columns)
    if len(cols) < 2:
        return df

    # 如果已经算过一次顺序，后面直接用缓存
    if _GLOBAL_COL_ORDER is not None:
        df_new = df[_GLOBAL_COL_ORDER].copy()
        # 后续调用也要对低 MI 特征做同样的伪删除
        if _GLOBAL_MI_FILTER_ENABLED:
            _apply_pseudo_delete(df_new)
        return df_new

    # ====== 第一次调用：真正计算顺序 ======

    start_time = time.time()

    time_col = cols[0]
    target_col = cols[-1]
    feat_cols = cols[1:-1]    # 中间特征列

    if len(feat_cols) == 0:
        _GLOBAL_COL_ORDER = cols
        _GLOBAL_MI_DROPPED_COLS = []
        return df

    # 数值特征 + target -> tensor
    val_cols = feat_cols + [target_col]         # 这些列对应 x 的最后一维
    x_np = df[val_cols].to_numpy(dtype=float)   # [N, D_val]
    x = torch.from_numpy(x_np).float()

    perm, _, mi, _ = build_perm_periodic(
        x,
        n_clusters=n_clusters,
        lags=lags,
        linkage=linkage,
        seed=seed,
    )  # perm: [D_val]

    perm_np = perm.cpu().numpy()
    mid_perm = perm_np[:-1]                      # 中间特征的新顺序（在 val_cols 的索引）

    # ====== 根据与 target 的 MI 选择“低信息特征”，但不删列，只做伪删除 ======
    dropped_cols = []
    if _GLOBAL_MI_FILTER_ENABLED:
        # mi 的最后一列是与 target 的互信息，前 D_val-1 行对应各个特征
        mi_to_target = mi[:-1, -1]  # 形状 [D_val-1]

        if mi_to_target.size > 0:
            # 找出 MI 低于阈值的特征索引（在 val_cols 里的位置）
            low_info_idx = np.where(mi_to_target < _MI_FILTER_THRESHOLD)[0]

            if low_info_idx.size == 0:
                # 如果没有任何特征低于阈值，则至少伪删除与 target MI 最小的一个特征
                min_idx = int(np.argmin(mi_to_target))
                drop_set = {min_idx}
            else:
                drop_set = set(low_info_idx.tolist())

            dropped_cols = [val_cols[i] for i in drop_set]

    # 根据排序后的索引得到特征列新顺序（保留所有特征，只是重排）
    feat_cols_new = [val_cols[i] for i in mid_perm]

    # 最终列顺序：时间 + 新特征顺序 + target
    new_cols = [time_col] + feat_cols_new + [target_col]

    # 先按新顺序重排
    df_new = df[new_cols].copy()

    # 记录伪删除的列名，并在当前 df 上应用伪删除
    _GLOBAL_MI_DROPPED_COLS = dropped_cols
    if _GLOBAL_MI_FILTER_ENABLED:
        _apply_pseudo_delete(df_new)

    end_time = time.time()

    log.Logger.log({
        'epoch': 'reorder',
        'cost_time': end_time - start_time,
    })

    print(f"重新排后的列顺序:{new_cols}，共耗时{end_time - start_time}s")
    if _GLOBAL_MI_FILTER_ENABLED and dropped_cols:
        print(f"根据与 target 的 MI 做伪删除处理的特征列: {dropped_cols}，模式: {_MI_FILTER_MODE}")

    # 缓存起来，后面 val/test 直接用
    _GLOBAL_COL_ORDER = new_cols

    return df_new
