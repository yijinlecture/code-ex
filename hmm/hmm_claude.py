# =========================
# 실제 제조 공정용 HMM 기반 상태 탐지 (노이지한 환경 대응)
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# 0) 실제 제조 공정과 유사한 복잡한 데이터 생성
# ------------------------------------------------------------------
np.random.seed(42)
freq = "30s"  # 30초 간격 (더 고빈도)
n_seg = 200   # 각 구간 길이 (30초 * 200 = 100분)
t = pd.date_range("2025-01-15 08:00", periods=n_seg * 5, freq=freq)

# 각 구간별 특성 (실제 제조 공정 시나리오)
def generate_realistic_manufacturing_data():
    # 1) 정상 운전 (약간의 주기적 변동 + 랜덤 노이즈)
    normal_base = 75
    t1 = np.linspace(0, 4*np.pi, n_seg)
    normal = (normal_base + 3*np.sin(t1) + 2*np.sin(3*t1) + 
              np.random.normal(0, 1.5, n_seg))
    
    # 2) 램프업 (설비 가동률 증가, 불안정함)
    ramp_start, ramp_end = 78, 145
    ramp_trend = np.linspace(ramp_start, ramp_end, n_seg)
    # 램프업 중 불안정성 (변동성 증가)
    instability = np.random.normal(0, 2.5, n_seg) * np.linspace(1, 2, n_seg)
    # 가끔 스파이크 발생
    spikes = np.random.choice([0, 8, -5], n_seg, p=[0.85, 0.08, 0.07])
    rising = ramp_trend + instability + spikes
    
    # 3) 고부하/비정상 운전 (높은 수준, 간헐적 스파이크)
    abnormal_base = 148
    # 주기적 진동 + 랜덤 스파이크
    t3 = np.linspace(0, 6*np.pi, n_seg)
    periodic_var = 4*np.sin(t3) + 2*np.sin(7*t3)
    random_spikes = np.where(np.random.random(n_seg) < 0.12, 
                            np.random.normal(15, 5, n_seg), 0)
    abnormal = abnormal_base + periodic_var + random_spikes + np.random.normal(0, 2, n_seg)
    
    # 4) 램프다운 (점진적 감소, 중간에 저항)
    down_trend = np.linspace(145, 82, n_seg)
    # 중간에 플래토 구간 (저항)
    resistance = np.where((np.arange(n_seg) > n_seg//3) & 
                         (np.arange(n_seg) < 2*n_seg//3), 8, 0)
    falling = down_trend + resistance + np.random.normal(0, 2, n_seg)
    
    # 5) 정상 복귀 (하지만 처음과는 약간 다른 패턴)
    normal2_base = 72  # 약간 다른 베이스라인
    t5 = np.linspace(0, 3*np.pi, n_seg)
    normal2 = (normal2_base + 2*np.sin(t5) + 
               np.random.normal(0, 1.2, n_seg))
    
    return np.concatenate([normal, rising, abnormal, falling, normal2])

R = generate_realistic_manufacturing_data()
df = pd.DataFrame({"time": t, "R": R}).set_index("time")

# ------------------------------------------------------------------
# 1) 고급 파생변수 생성 (제조 공정에 특화)
# ------------------------------------------------------------------
def build_manufacturing_features(df: pd.DataFrame, value_col="R"):
    """
    실제 제조 공정을 위한 고급 특징 추출
    - 다양한 시간 윈도우
    - 노이즈 필터링
    - 제조 공정 특화 지표들
    """
    f = df.copy()
    s = f[value_col].astype(float)
    
    # 노이즈 필터링 (이동평균)
    f["R_smooth"] = s.rolling(5, center=True).mean().fillna(method='bfill').fillna(method='ffill')
    
    # 1) 단기 변화량 (1분, 3분)
    f["dR_1min"] = s.diff(2).fillna(0.0)      # 1분 차이
    f["dR_3min"] = s.diff(6).fillna(0.0)      # 3분 차이
    
    # 2) 다양한 시간 윈도우 추세 (5분, 15분, 30분)
    for win, name in [(10, "5min"), (30, "15min"), (60, "30min")]:
        f[f"trend_{name}"] = (s.rolling(win)
                             .apply(lambda x: x[-1] - x[0] if len(x) >= 2 else 0, raw=True)
                             .fillna(0))
    
    # 3) 변동성 지표들
    f["volatility_5min"] = s.rolling(10).std().fillna(0)
    f["volatility_15min"] = s.rolling(30).std().fillna(0)
    
    # 4) 수준 지표 (이동평균 대비)
    f["level_vs_5min"] = s - s.rolling(10).mean().fillna(s)
    f["level_vs_30min"] = s - s.rolling(60).mean().fillna(s)
    
    # 5) 스파이크 탐지 (Z-score 기반)
    rolling_mean = s.rolling(10).mean().fillna(s)
    rolling_std = s.rolling(10).std().fillna(1)
    f["spike_score"] = np.abs((s - rolling_mean) / (rolling_std + 1e-6))
    
    # 6) 가속도 (변화량의 변화량)
    f["acceleration"] = f["dR_1min"].diff().fillna(0)
    
    return f

df_feat = build_manufacturing_features(df, value_col="R")

# ------------------------------------------------------------------
# 2) Robust 스케일링 (아웃라이어에 강건)
# ------------------------------------------------------------------
# 제조 공정은 아웃라이어가 많으므로 RobustScaler 사용
feat_cols = [
    "R", "dR_1min", "dR_3min", 
    "trend_5min", "trend_15min", "trend_30min",
    "volatility_5min", "volatility_15min",
    "level_vs_5min", "level_vs_30min",
    "spike_score", "acceleration"
]

scaler = RobustScaler()  # 중간값과 IQR 기반 스케일링
X = scaler.fit_transform(df_feat[feat_cols].values)

# ------------------------------------------------------------------
# 3) 적응적 HMM 설정 (실제 환경용)
# ------------------------------------------------------------------
# 더 많은 반복, 더 유연한 공분산 구조
hmm = GaussianHMM(
    n_components=4, 
    covariance_type="full",     # 변수 간 상관관계 고려
    n_iter=500,                 # 충분한 학습
    tol=1e-4,                   # 수렴 기준 완화
    random_state=42,
    algorithm="viterbi"         # 안정적인 디코딩
)

# 초기 전이행렬 가이드 (제조 공정 특성 반영)
# 정상 상태에서 머무르는 확률이 높고, 급격한 변화는 제한
initial_transmat = np.array([
    [0.85, 0.10, 0.03, 0.02],  # 정상 → 대부분 정상 유지
    [0.05, 0.80, 0.12, 0.03],  # 상승 → 상승 지속 or 비정상으로
    [0.02, 0.05, 0.85, 0.08],  # 비정상 → 대부분 비정상 유지
    [0.08, 0.02, 0.05, 0.85],  # 하강 → 하강 지속 or 정상으로
])

hmm.startprob_ = np.array([0.7, 0.1, 0.1, 0.1])
hmm.transmat_ = initial_transmat
hmm.init_params = 'mc'  # 평균과 공분산만 학습

hmm.fit(X)
states = hmm.predict(X)
df_feat["hmm_state"] = states

# ------------------------------------------------------------------
# 4) 실제 환경용 라벨링 함수 (더 정교한 임계값)
# ------------------------------------------------------------------
def manufacturing_state_labeling(
    model: GaussianHMM,
    feat_names,
    spike_thr=1.5,      # 스파이크 임계값 (낮춤)
    trend_thr=0.3,      # 추세 임계값 (낮춤)
    volatility_thr=0.5, # 변동성 임계값
    level_thr=0.6       # 수준 임계값 (낮춤)
):
    means = model.means_
    
    # 인덱스 매핑
    idx_map = {name: i for i, name in enumerate(feat_names)}
    
    def classify_state(state_id: int) -> str:
        mean_vec = means[state_id]
        
        # 주요 지표들 추출
        r_level = mean_vec[idx_map["R"]]
        spike = mean_vec[idx_map["spike_score"]]
        vol_5min = mean_vec[idx_map["volatility_5min"]]
        trend_5min = mean_vec[idx_map["trend_5min"]]
        trend_15min = mean_vec[idx_map["trend_15min"]]
        
        # 1) 스파이크가 많거나 높은 수준 → 비정상
        if (spike > spike_thr) or (r_level > level_thr and vol_5min > volatility_thr):
            return "비정상"
        
        # 2) 명확한 상승 추세
        if (trend_5min > trend_thr) or (trend_15min > trend_thr * 0.8):
            return "상승"
        
        # 3) 명확한 하강 추세  
        if (trend_5min < -trend_thr) or (trend_15min < -trend_thr * 0.8):
            return "하강"
        
        # 4) 그 외 → 정상
        return "정상"
    
    return classify_state

label_fn = manufacturing_state_labeling(
    hmm, 
    feat_names=feat_cols,
    spike_thr=1.2,      # 실제 환경에서는 더 민감하게
    trend_thr=0.25,     # 작은 변화도 감지
    volatility_thr=0.4,
    level_thr=0.5
)

df_feat["trend4"] = df_feat["hmm_state"].apply(label_fn)

# ------------------------------------------------------------------
# 5) 적응적 후처리 (실제 환경용)
# ------------------------------------------------------------------
def adaptive_smoothing(labels: pd.Series, min_normal=15, min_abnormal=8):
    """
    상태별로 다른 최소 지속 시간 적용
    - 정상: 더 긴 최소 지속 시간 (잘못된 알람 방지)
    - 비정상: 짧은 지속 시간도 허용 (빠른 감지)
    """
    vals = labels.to_numpy()
    n = len(vals)
    
    i = 0
    while i < n:
        j = i
        while j < n and vals[j] == vals[i]:
            j += 1
        
        current_label = vals[i]
        duration = j - i
        
        # 상태별 최소 지속 시간 체크
        min_len = min_normal if current_label == "정상" else min_abnormal
        
        if duration < min_len:
            # 이웃 구간으로 흡수
            left = vals[i-1] if i > 0 else None
            right = vals[j] if j < n else None
            
            # 비정상 상태는 우선권 부여
            if current_label == "비정상":
                if duration >= min_abnormal // 2:  # 절반 이상이면 유지
                    i = j
                    continue
            
            fill_val = right if right is not None else left
            if fill_val is not None:
                vals[i:j] = fill_val
        
        i = j
    
    return pd.Series(vals, index=labels.index)

# 적응적 후처리 적용
df_feat["trend4_smooth"] = adaptive_smoothing(df_feat["trend4"], 
                                             min_normal=20,    # 10분
                                             min_abnormal=6)   # 3분

# ------------------------------------------------------------------
# 6) 고급 시각화 (실제 환경용)
# ------------------------------------------------------------------
def plot_manufacturing_analysis(df, title="제조 공정 상태 모니터링"):
    plt.style.use("dark_background")
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    # 색상 맵
    colors = {
        "정상": "#4CAF50",      # 초록
        "상승": "#FF9800",      # 주황  
        "비정상": "#F44336",    # 빨강
        "하강": "#2196F3"       # 파랑
    }
    
    # 1) 원본 데이터와 스무스 버전
    ax1 = axes[0]
    ax1.plot(df.index, df["R"], alpha=0.6, color="gray", linewidth=0.8, label="원본")
    ax1.plot(df.index, df["R_smooth"], color="white", linewidth=1.5, label="스무스")
    
    for label, color in colors.items():
        mask = df["trend4_smooth"] == label
        if mask.any():
            ax1.scatter(df.index[mask], df["R"][mask], 
                       c=color, s=15, alpha=0.8, label=label)
    
    ax1.set_ylabel("공정 파라미터 (R)")
    ax1.set_title(f"{title} - 상태별 분류")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2) 변동성과 스파이크 점수
    ax2 = axes[1]
    ax2.plot(df.index, df["volatility_15min"], color="cyan", alpha=0.7, label="변동성(15분)")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(df.index, df["spike_score"], color="yellow", alpha=0.7, label="스파이크 점수")
    
    ax2.set_ylabel("변동성", color="cyan")
    ax2_twin.set_ylabel("스파이크 점수", color="yellow")
    ax2.tick_params(axis='y', labelcolor="cyan")
    ax2_twin.tick_params(axis='y', labelcolor="yellow")
    ax2.grid(True, alpha=0.3)
    
    # 3) 추세 지표들
    ax3 = axes[2]
    ax3.plot(df.index, df["trend_5min"], color="orange", alpha=0.8, label="5분 추세")
    ax3.plot(df.index, df["trend_15min"], color="lightblue", alpha=0.8, label="15분 추세")
    ax3.axhline(y=0, color="white", linestyle="--", alpha=0.5)
    
    ax3.set_ylabel("추세 지표")
    ax3.set_xlabel("시간")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

plot_manufacturing_analysis(df_feat)

# ------------------------------------------------------------------
# 7) 성능 진단 및 조정 가이드
# ------------------------------------------------------------------
print("=" * 60)
print("제조 공정 HMM 상태 탐지 결과 분석")
print("=" * 60)

# 기본 통계
print("\n[1] 라벨 분포:")
label_dist = df_feat["trend4_smooth"].value_counts()
for label, count in label_dist.items():
    pct = count / len(df_feat) * 100
    print(f"  {label}: {count}개 ({pct:.1f}%)")

# 상태별 특성 요약
print("\n[2] 상태별 주요 지표 평균:")
summary_cols = ["R", "volatility_15min", "spike_score", "trend_15min"]
state_summary = df_feat.groupby("trend4_smooth")[summary_cols].mean()
print(state_summary.round(2))

# 전이 분석
print("\n[3] 상태 전이 분석:")
transitions = []
prev_state = None
for state in df_feat["trend4_smooth"]:
    if prev_state is not None and prev_state != state:
        transitions.append(f"{prev_state}→{state}")
    prev_state = state

trans_counts = pd.Series(transitions).value_counts()
print("주요 전이 패턴:")
for trans, count in trans_counts.head(6).items():
    print(f"  {trans}: {count}번")

print("\n[4] 튜닝 가이드:")
print("만약 결과가 만족스럽지 않다면:")
print("- 너무 많은 '비정상' → spike_thr, level_thr 값을 높이세요")
print("- 너무 적은 '비정상' → spike_thr, level_thr 값을 낮추세요") 
print("- 상승/하강을 못 잡음 → trend_thr 값을 낮추세요")
print("- 너무 민감함 → min_normal, min_abnormal 값을 높이세요")