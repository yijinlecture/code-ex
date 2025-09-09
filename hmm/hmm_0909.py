# =========================
# HMM 기반 4라벨(정상/상승/비정상/하강) 전체 예시
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------
# 0) 더미 시계열 생성 (정상 → 상승 → 비정상 → 하강 → 정상)
# ------------------------------------------------------------------
np.random.seed(7)
freq = "5min"
n_seg = 80  # 각 구간 길이 (5분 간격 * 80 = 400분)
t = pd.date_range("2025-08-18", periods=n_seg * 5, freq=freq)

# 구간별 생성
low_plateau = np.random.normal(50, 2.0, n_seg)  # 정상(낮은 plateau)
rising = np.linspace(60, 148, n_seg) + np.random.normal(0, 1, n_seg)  # 상승(램프업)
high_plateau = np.random.normal(150, 2.0, n_seg)  # 비정상(높은 plateau)
falling = np.linspace(148, 60, n_seg) + np.random.normal(0, 1, n_seg)  # 하강(램프다운)
low_plateau2 = np.random.normal(50, 2.0, n_seg)  # 정상 복귀

R = np.concatenate([low_plateau, rising, high_plateau, falling, low_plateau2])
df = pd.DataFrame({"time": t, "R": R}).set_index("time")


# ------------------------------------------------------------------
# 1) 파생변수 생성(수준/방향/추세/변동성)
# ------------------------------------------------------------------
def build_features(df: pd.DataFrame, value_col="R", win=12) -> pd.DataFrame:
    """
    - dR: 직전 대비 변화량 (상승/하강 민감)
    - cum_inc: 창 구간 시작-끝 차이 (완만 추세)
    - roll_std: 창 표준편차 (변동성)
    """
    f = df.copy()
    s = f[value_col].astype(float)
    f["dR"] = s.diff().fillna(0.0)
    f["cum_inc"] = (
        s.rolling(win).apply(lambda x: x[-1] - x[0], raw=True).bfill().ffill()
    )
    f["roll_std"] = s.rolling(win).std().bfill().ffill()
    return f


df_feat = build_features(df, value_col="R", win=12)  # 12*5분 ≈ 1시간 창

# ------------------------------------------------------------------
# 2) 표준화 (HMM의 공정성 확보)
# ------------------------------------------------------------------
feat_cols = ["R", "dR", "cum_inc", "roll_std"]
scaler = StandardScaler()
X = scaler.fit_transform(df_feat[feat_cols].values)

# ------------------------------------------------------------------
# 3) HMM 학습 + 상태 디코딩
# ------------------------------------------------------------------
hmm = GaussianHMM(n_components=4, covariance_type="full", n_iter=300, random_state=42)
hmm.fit(X)
states = hmm.predict(X)
df_feat["hmm_state"] = states

# (선택) 전이 구조를 가이드하고 싶다면 아래처럼 초기 전이행렬/시작확률을 설정하고 평균/공분산만 학습:
# hmm = GaussianHMM(n_components=4, covariance_type="full", n_iter=200, random_state=42)
# hmm.startprob_ = np.array([1.0, 0.0, 0.0, 0.0])
# hmm.transmat_  = np.array([
#     [0.95, 0.05, 0.00, 0.00],  # 정상 → 상승
#     [0.00, 0.95, 0.05, 0.00],  # 상승 → 비정상
#     [0.00, 0.00, 0.95, 0.05],  # 비정상 → 하강
#     [0.05, 0.00, 0.00, 0.95],  # 하강 → 정상
# ])
# hmm.init_params = 'mc'  # m(평균), c(공분산)만 업데이트
# hmm.fit(X)
# df_feat["hmm_state"] = hmm.predict(X)


# ------------------------------------------------------------------
# 4) 상태 → 4라벨 매핑 함수 (정상/상승/비정상/하강)
#    * 상태별 평균 피처(표준화 스케일)를 이용한 휴리스틱
# ------------------------------------------------------------------
def state_to_four_labels(
    model: GaussianHMM,
    feat_names=("R", "dR", "cum_inc", "roll_std"),
    thr_up=0.35,
    thr_down=-0.35,
    high_level_thr=0.8,
    low_vol_thr=0.1,
):
    """
    - thr_up/down : dR 또는 cum_inc 기준의 상승/하강 임계값 (z-score)
    - high_level_thr : R이 충분히 높다고 보는 기준(비정상 plateau)
    - low_vol_thr    : 변동성 낮음(plateau) 판단 기준
    """
    means = model.means_
    iR, iDR, iCI, iRS = [feat_names.index(n) for n in feat_names]

    def lab(state_id: int) -> str:
        r = means[state_id, iR]
        dr = means[state_id, iDR]
        ci = means[state_id, iCI]
        rs = means[state_id, iRS]

        # 높은 수준 + 낮은 변동성 → 비정상 plateau
        if (
            (r >= high_level_thr)
            and (abs(dr) < 0.2)
            and (abs(ci) < 0.2)
            and (abs(rs) <= low_vol_thr)
        ):
            return "비정상"

        # 상승/하강(즉각 기울기 or 누적변화 둘 중 하나만 넘으면)
        if (dr >= thr_up) or (ci >= thr_up):
            return "상승"
        if (dr <= thr_down) or (ci <= thr_down):
            return "하강"

        # 그 외 평탄 → 정상
        return "정상"

    return lab


label_fn4 = state_to_four_labels(
    hmm,
    feat_names=tuple(feat_cols),
    thr_up=0.35,
    thr_down=-0.35,
    high_level_thr=0.8,
    low_vol_thr=0.1,
)

df_feat["trend4"] = df_feat["hmm_state"].apply(label_fn4)


# ------------------------------------------------------------------
# 5) 후처리 (짧은 구간 흡수 + 양옆 병합) → 깔끔한 세그먼트
# - 짧은 구간 제거(30분 미만의 짧은 변화는 노이즈로 보고 제거
# - 이웃 구한 병합: A-B-A패턴을 A-A-A로 부드럽게 만들기
# ------------------------------------------------------------------
def enforce_min_run(labels: pd.Series, min_len=6):
    vals = labels.to_numpy()
    n = len(vals)
    s = 0
    while s < n:
        e = s
        while e < n and vals[e] == vals[s]:
            e += 1
        if (e - s) < min_len:  # 최소 길이 미만이면 이웃 구간으로 흡수
            left = vals[s - 1] if s - 1 >= 0 else None
            right = vals[e] if e < n else None
            fill = right if right is not None else left
            if fill is not None:
                vals[s:e] = fill
        s = e
    return pd.Series(vals, index=labels.index)


def merge_neighbors(labels: pd.Series):
    vals = labels.to_numpy()
    n = len(vals)
    for i in range(1, n - 1):
        if vals[i - 1] == vals[i + 1] and vals[i] != vals[i - 1]:
            vals[i] = vals[i - 1]
    return pd.Series(vals, index=labels.index)


# 5분 간격 * 6 = 30분 미만 스파이크 제거 예시
df_feat["trend4"] = enforce_min_run(df_feat["trend4"], min_len=6)
df_feat["trend4"] = merge_neighbors(df_feat["trend4"])


# ------------------------------------------------------------------
# 6) 시각화 (각 데이터 점을 라벨별 색상으로 표시)
# ------------------------------------------------------------------
def plot_trend4(df, value_col="R", trend_col="trend4", title=None, dark=True):
    if dark:
        plt.style.use("dark_background")
    cmap = {
        "정상": "cornflowerblue",
        "상승": "orange",
        "비정상": "tomato",
        "하강": "mediumseagreen",
    }

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(
        df.index,
        df[value_col],
        lw=1.2,
        alpha=0.7,
        color="white" if dark else "black",
        ls="--",
        label="Original R-Value",
    )

    for lab, c in cmap.items():
        part = df[df[trend_col] == lab]
        ax.scatter(part.index, part[value_col], s=24, alpha=0.95, c=c, label=lab)

    ax.set_ylabel("R")
    ax.set_xlabel("시간")
    ax.set_title(title or "HMM 기반 4라벨(정상/상승/비정상/하강)")
    ax.legend(loc="best", framealpha=0.6)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


plot_trend4(
    df_feat,
    value_col="R",
    trend_col="trend4",
    title="HMM을 이용한 시계열 상태 탐지 최종 결과 (K-Means보다 안정적)",
)

# ------------------------------------------------------------------
# 7) 간단 진단 리포트: 라벨 분포 / 상태별 요약
# ------------------------------------------------------------------
print("\n[라벨 분포]")
print(df_feat["trend4"].value_counts())

print("\n[라벨별 R 통계 요약]")
print(df_feat.groupby("trend4")["R"].describe()[["count", "mean", "std", "min", "max"]])
