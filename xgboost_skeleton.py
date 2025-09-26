import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 예시 데이터 생성 (x값 10개, y값 1개)
np.random.seed(42)
X = np.random.rand(100, 10)  # 100개의 샘플, 10개의 x값
y = np.random.rand(100) * 100  # 0~100 사이의 y값

# 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# XGBoost 회귀 모델 설정
model = xgb.XGBRegressor(
    n_estimators=200,      # 트리 개수
    learning_rate=0.1,     # 학습률
    max_depth=5,           # 트리 깊이
    subsample=0.8,         # 샘플 비율
    colsample_bytree=0.8,  # 피처 샘플링 비율
    random_state=42
)

# 모델 학습
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R² Score:", r2)

# 피처 중요도 확인
import matplotlib.pyplot as plt
xgb.plot_importance(model)
plt.show()
