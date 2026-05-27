import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 1. 데이터 로드 및 전처리
url = "C:\\Users\\user\\Desktop\\대학교\\4학년\\의로데이터분석\\프로젝트\\diabetes.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=column_names, header=None)
df = df.apply(pd.to_numeric, errors='coerce')

cols_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_fix] = df[cols_fix].replace(0, np.nan)
df.fillna(df.median(numeric_only=True), inplace=True)

# 2. 데이터 분할
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. 스케일링
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 고도화된 SVM 학습 (RBF 커널 + 클래스 가중치)
# C와 gamma는 튜닝의 핵심 포인트입니다.
model = SVC(
    kernel='rbf', 
    C=1.0, 
    gamma='scale', 
    class_weight='balanced', 
    probability=True, 
    random_state=42
)
model.fit(X_train_scaled, y_train)

# 5. 결과 확인
y_pred = model.predict(X_test_scaled)
print(f"개선된 모델 정확도: {accuracy_score(y_test, y_pred):.4f}")
print("\n[개선된 분류 보고서]\n", classification_report(y_test, y_pred))

# 6. 모델 및 스케일러 업데이트 저장 (절대 경로 지정)
save_dir = "C:\\Users\\user\\Desktop\\대학교\\4학년\\의로데이터분석\\프로젝트\\"

with open(save_dir + 'model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
with open(save_dir + 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"🎉 파일이 다음 경로에 정확히 저장되었습니다: {save_dir}")