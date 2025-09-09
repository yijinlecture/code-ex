import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib

# Windows 환경에서 한글 폰트 깨짐 방지
from matplotlib import font_manager, rc

try:
    font_path = "c:/Windows/Fonts/malgun.ttf"
    font = font_manager.FontProperties(fname=font_path).get_name()
    rc("font", family=font)
except:
    print("Malgun Gothic font not found... skipping font setup")
plt.rcParams["axes.unicode_minus"] = False


def label_trend_with_sliding_window(
    series: pd.Series, window_size: int, slope_threshold: float
) -> list:
    """
    슬라이딩 윈도우와 선형 회귀 기울기를 사용해 추세를 라벨링합니다.

    :param series: 시계열 데이터 (Pandas Series)
    :param window_size: 추세를 계산할 창문의 크기
    :param slope_threshold: 상승/하강을 판단할 기울기 임계값
    :return: 각 포인트에 대한 라벨 리스트
    """
    labels = []
    # 윈도우를 한 칸씩 이동하며 전체 데이터를 순회합니다.
    for i in range(len(series)):
        # 윈도우가 데이터 범위를 벗어나지 않도록 조정
        start = max(0, i - window_size // 2)
        end = min(len(series), i + window_size // 2 + 1)

        window = series[start:end]

        # 윈도우 내 데이터로 선형 회귀를 수행하여 기울기를 계산
        x = np.arange(len(window))
        # np.polyfit(x, y, 1)[0]은 1차식(직선)의 기울기를 반환합니다.
        slope = np.polyfit(x, window.values, 1)[0]

        # 기울기와 임계값을 비교하여 라벨 결정
        if slope > slope_threshold:
            labels.append("상승")
        elif slope < -slope_threshold:
            labels.append("하강")
        else:
            labels.append("평탄")

    return labels


# 1. 샘플 데이터 생성 (HMM 예제와 유사하게)
np.random.seed(42)
data_flat1 = np.random.randn(100) * 0.5 + 50
data_rise = np.linspace(50, 80, 50) + np.random.randn(50) * 0.4
data_flat2 = np.random.randn(80) * 0.8 + 80
data_fall = np.linspace(80, 50, 70) + np.random.randn(70) * 0.4
data_flat3 = np.random.randn(100) * 0.5 + 50
waveform_data = pd.Series(
    np.concatenate([data_flat1, data_rise, data_flat2, data_fall, data_flat3])
)

# 2. 파라미터 설정 및 라벨링 실행
WINDOW_SIZE = 10  # 창문 크기 (데이터 10개)
SLOPE_THRESHOLD = 0.3  # 기울기 임계값 (이 값보다 크면 상승/작으면 하강)

labels = label_trend_with_sliding_window(waveform_data, WINDOW_SIZE, SLOPE_THRESHOLD)
df = pd.DataFrame({"value": waveform_data, "label": labels})

# 3. 시각화
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(16, 6))

colors = {"평탄": "green", "상승": "red", "하강": "blue"}

# 원본 파형 그리기
ax.plot(df.index, df["value"], color="white", linewidth=2, label="Waveform Data")

# 라벨에 따라 배경색 칠하기
for i in range(len(df) - 1):
    ax.axvspan(
        df.index[i],
        df.index[i + 1],
        alpha=0.5,
        color=colors[df["label"].iloc[i]],
        ec="none",
    )

ax.set_title("Rule-Based Trend Labeling with Sliding Window", fontsize=16)
handles = [
    plt.Rectangle((0, 0), 1, 1, color=colors[label], alpha=0.5) for label in colors
]
ax.legend(handles, colors.keys())
plt.show()
