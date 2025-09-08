import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.api import VAR

# --------------------------------------
# 1. 데이터 준비 (예시용 시뮬레이션 데이터)
# --------------------------------------
np.random.seed(42)
n = 300
time = pd.date_range("2024-01-01", periods=n, freq="D")

# 변수 생성: y는 a와 fluoride의 영향을 받는다고 가정
a = np.random.normal(0, 1, n).cumsum()
fluoride = np.random.normal(0, 1, n).cumsum()
x = np.random.normal(0, 1, n).cumsum()
y = 0.4 * np.roll(a, 2) + 0.6 * np.roll(fluoride, 5) + np.random.normal(0, 1, n)

df = pd.DataFrame({"y": y, "a": a, "x": x, "fluoride": fluoride}, index=time)

print(df.head())

# --------------------------------------
# 2. 정상성 확인 (ADF Test)
# --------------------------------------
def adf_test(series, signif=0.05):
    result = adfuller(series.dropna(), autolag="AIC")
    return result[1] < signif

print("\n[정상성 검정]")
for col in df.columns:
    print(f"{col}: {'정상' if adf_test(df[col]) else '비정상'}")

# 비정상 -> 차분
df_diff = df.diff().dropna()

# --------------------------------------
# 3. Granger Causality Test
# --------------------------------------
print("\n[Granger Causality Test: a → y]")
granger_result = grangercausalitytests(df_diff[["y", "a"]], maxlag=6, verbose=True)

# --------------------------------------
# 4. VAR 모델 적합
# --------------------------------------
model = VAR(df_diff)
order = model.select_order(maxlags=6)
print("\n[VAR 차수 선택]\n", order.summary())

var_res = model.fit(order.aic)
print(var_res.summary())

# --------------------------------------
# 5. Impulse Response Function (IRF)
# --------------------------------------
irf = var_res.irf(10)  # 10-step 반응
irf.plot(orth=True)
plt.suptitle("Impulse Response Function", y=1.02)
plt.show()

# --------------------------------------
# 6. Forecast Error Variance Decomposition (FEVD)
# --------------------------------------
fevd = var_res.fevd(10)
print("\n[Forecast Error Variance Decomposition]\n")
print(fevd.summary())
