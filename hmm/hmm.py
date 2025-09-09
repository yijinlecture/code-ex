import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# HMM(가우시안) 모델: 상태별 관측값이 정규분포라고 가정
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# =========================================
# 0) 피처 만들기: 시계열에서 파생변수 생성
# =========================================
def build_features(df: pd.DataFrame, value_col: str = "R", win: int = 6) -> pd.DataFrame:
    """
    [개념] 상태(정상/전이/비정상)는 '수준'뿐 아니라 '방향/변동성'으로도 구분
    [역할] HMM이 상태를 더 잘 나누도록, 수준과 추세를 보여주는 파생변수를 만들어줌

    - dR(1차차분): 직전 대비 변화량(즉각적인 상승/하락 감지)
        - dR > 0 : 상승 / dR < 0 : 하락
    - cum_inc(누적상승): 최근 win 구간의 시작-끝 차이(완만한 추세 감지)
        - "오르는 중/내리는 중"을 잡는데 특화
    - roll_std(변동성): 최근 win 구간의 표준편차(안정/요동 구분)
        - 전이,불안정 구간에서는 커지고, 안정(정상/비정상 유지)구간에서는 작아지는 경향이 있음.
        - 상태 유지와 전이를 구분하는데 도움이 될 수 있음.
    """
    fdf = df.copy()
    s = fdf[value_col].astype(float)

    # [방향] 바로 직전 대비 얼마나 변했는지 (전이 시점에 민감)
    fdf["dR"] = s.diff().fillna(0.0)

    # [추세] 최근 win개 구간에서 얼마나 누적 상승(또는 하락)했는지
    fdf["cum_inc"] = s.rolling(win).apply(lambda x: x[-1] - x[0], raw=True)
    # rolling 초반 NaN 보정(앞/뒤 값으로 메움) → 초반 몇 포인트는 참고치로 사용
    fdf["cum_inc"] = fdf["cum_inc"].bfill().ffill()

    # [변동성] 최근 구간이 얼마나 흔들렸는지
    fdf["roll_std"] = s.rolling(win).std()
    fdf["roll_std"] = fdf["roll_std"].bfill().ffill()

    # ⟨여기⟩ win 크기: 6(=30분), 12(=60분) 등 현장 응답속도에 맞춰 조정 
    # 
    return fdf


def standardize_features(fdf: pd.DataFrame, feat_cols=("R", "dR", "cum_inc", "roll_std")):
    """
    [개념] HMM은 '분포'를 학습하므로, 피처 스케일이 제각각이면 특정 피처만 과도하게 반영됨
    [역할] 모든 피처를 평균0/분산1로 맞춰 공정하게 학습되게 함
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(fdf[list(feat_cols)].values)
    # ⟨여기⟩ feat_cols에 조작변수 요약(최근평균, 기울기 등)을 추가해 성능 향상 시도 가능할듯.
    return X, scaler


# =========================================
# 1) HMM 학습(EM) + 상태 추론(Viterbi)
# =========================================
def fit_hmm(X: np.ndarray, n_states: int = 4, cov_type: str = "full",
            n_iter: int = 200, random_state: int = 42) -> GaussianHMM:
    """
    [개념] '상태/전이/관측분포'를 데이터에 맞게 추정.
    [역할] 라벨 없이도(비지도) 상태 구조를 찾아냄.
    """
    model = GaussianHMM(
        n_components=n_states,      # 숨겨진 상태 수(2~4 추천). 많을수록 세밀하지만 불안정 가능
        covariance_type=cov_type,   # "full": 유연 / "diag": 빠르고 안정적(과적합 방지)
        n_iter=n_iter,              # 반복 횟수(수렴 경고나면 늘리기)
        random_state=random_state
    )
    model.fit(X)
    # ⟨여기⟩ cov_type="diag" 시도, n_states 2→3→4 탐색, n_iter 300~500
    return model


def decode_states(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """
    [개념] Viterbi: 전체 시계열에서 가장 그럴듯한 상태열(시점별 상태)을 찾는 동적계획법.
    [역할] 각 시점에 상태 번호(0..n-1)를 할당.
    """
    return model.predict(X)


# =========================================
# 2) 상태 번호 → 사람이 읽는 라벨(휴리스틱)
# =========================================
def map_states_to_labels(model: GaussianHMM,
                         feat_names=("R", "dR", "cum_inc", "roll_std"),
                         verbose: bool = True):
    """
    [개념] HMM이 찾은 상태는 '숫자'일 뿐. 상태별 평균 피처(표준화 스케일)를 보고 해석해야 함.
    [역할] 상태별 평균이 높은/오르는/내리는/요동치는지 등을 기준으로 라벨 부여.
    """
    means = model.means_  # 표준화 스케일 상의 평균들

    if verbose:
        print("=== 상태별 평균(표준화) ===")
        for k, mu in enumerate(means):
            print(f"state {k}: ", {n: round(v, 3) for n, v in zip(feat_names, mu)})

    def label_fn(state_id: int) -> str:
        r, dr, ci, rs = means[state_id]
        # ⟨여기⟩ 임계값(0.8, 0.6, 0.5)은 예시. 아래 '상태별 통계'를 보고 현장 기준으로 조정 가능할듯!
        if (r > 0.8) and (abs(dr) < 0.2):   # 수준 높고 안정적 → 비정상 유지
            return "비정상"
        if (ci > 0.6) or (dr > 0.5):        # 상승 추세 → 정상→비정상
            return "정상→비정상"
        if (ci < -0.6) or (dr < -0.5):      # 하강 추세 → 비정상→정상
            return "비정상→정상"
        return "정상"

    return label_fn


# =========================================
# 3) 시각화: 상태 라벨을 색으로 표시
# =========================================
def plot_states(df: pd.DataFrame, value_col: str = "R",
                state_col: str = "hmm_state", label_col: str = "hmm_label",
                title: str = "HMM 시계열 상태 구분", colors=None):
    """
    [개념] 눈으로 상태 전개(정상→전이→비정상→전이→정상)를 확인하는 것이 중요함
    [역할] 원 시계열 위에 상태별 색상을 찍어 해석을 도와줌
    """
    if colors is None:
        colors = {"정상": "green", "정상→비정상": "orange", "비정상": "red", "비정상→정상": "blue"}

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df.index, df[value_col], lw=1, alpha=0.5, color="gray", label=value_col)

    for lab, c in colors.items():
        part = df[df[label_col] == lab]
        ax.scatter(part.index, part[value_col], s=10, color=c, label=lab)

    ax.set_title(title)
    ax.set_ylabel(value_col)
    ax.legend(loc="upper left", ncol=len(colors))
    plt.tight_layout()
    plt.show()


# =========================================
# 4) 실행 예시(샘플 시계열)
# =========================================
if __name__ == "__main__":
    # (A) 샘플: 정상 → 완만 상승(전이 up) → 비정상 → 완만 하강(전이 down)
    np.random.seed(7)
    time = pd.date_range("2025-01-01", periods=240, freq="5min")
    R = np.concatenate([
        np.random.normal(100, 2, 60),
        np.linspace(100, 128, 60) + np.random.normal(0, 1, 60),
        np.random.normal(130, 2.5, 60),
        np.linspace(130, 102, 60) + np.random.normal(0, 1, 60),
    ])
    df = pd.DataFrame({"time": time, "R": R}).set_index("time")

    # (B) 피처 생성 + 표준화
    df_feat = build_features(df, value_col="R", win=6)  # ⟨여기⟩ 6=30분. 데이터 주기·현장감에 맞춰 조정
    X, scaler = standardize_features(df_feat)

    # (C) HMM 학습
    hmm = fit_hmm(X, n_states=4, cov_type="full", n_iter=200, random_state=42)
    # ⟨여기⟩ n_states: 2→3→4 순으로 늘려 비교
    # ⟨여기⟩ cov_type="diag"로 바꾸면 속도·안정성 up
    # ⟨여기⟩ 수렴 경고 뜨면 n_iter=200~500

    # (D) 상태 추론
    df_feat["hmm_state"] = decode_states(hmm, X)

    # (E) 숫자 상태 → 라벨
    label_fn = map_states_to_labels(hmm, verbose=True)
    df_feat["hmm_label"] = df_feat["hmm_state"].apply(label_fn)
    # ⟨여기⟩ 라벨 임계값은 '상태별 평균'과 아래 '상태별 통계'를 보며 미세 조정

    # (F) 시각화
    plot_states(df_feat, value_col="R", state_col="hmm_state", label_col="hmm_label")

    # (G) 상태별 통계(라벨 기준 검증)
    print("\n=== 상태별 피처 요약 ===")
    print(df_feat.groupby("hmm_label")[["R", "dR", "cum_inc", "roll_std"]].describe())
