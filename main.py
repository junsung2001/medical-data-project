from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter

# 환경 변수 로드
load_dotenv()

# 1. FastAPI 앱 초기화
app = FastAPI(
    title="당뇨병 위험도 예측 및 AI 상담 API 서버",
    description="Pima Indians Diabetes Dataset 기반 SVM 예측 및 OpenAI 상담 서버",
    version="1.3.0"
)

# Prometheus 모니터링 설정
instrumentator = Instrumentator().instrument(app)

# 커스텀 메트릭 정의 (AI 토큰 사용량 추적용 Counter)
AI_TOKEN_USAGE_COUNTER = Counter(
    "ai_counseling_tokens_total", 
    "Total number of tokens used in AI counseling requests"
)

@app.on_event("startup")
async def startup():
    instrumentator.expose(app)

# CORS 설정 (프론트엔드 연동을 위해 필요)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 비용 방어 및 토큰 관리 설정
MAX_TOKEN_LIMIT = 4000  # 단일 요청 최대 토큰 제한
TOKEN_USAGE_TRACKER = {"total_tokens": 0}

def count_tokens(text: str, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# 2. 학습된 ML 모델 및 스케일러 로드
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    print("🎉 머신러닝 모델 및 스케일러 로드 성공!")
except FileNotFoundError:
    print("❌ model.pkl 또는 scaler.pkl 파일을 찾을 수 없습니다. 먼저 모델을 학습시켜 주세요.")
    model = None
    scaler = None

# 3. 데이터 입력 포맷 정의
class DiabetesInput(BaseModel):
    pregnancies: int = Field(..., description="임신 횟수", ge=0)
    glucose: float = Field(..., description="공복 혈당 수치", ge=0)
    blood_pressure: float = Field(..., description="혈압 (이완기)", ge=0)
    skin_thickness: float = Field(..., description="피부 두께", ge=0)
    insulin: float = Field(..., description="인슐린 수치", ge=0)
    bmi: float = Field(..., description="체질량지수 (BMI)", ge=0)
    diabetes_pedigree: float = Field(..., description="당뇨 직계 가족력 유전 지수", ge=0)
    age: int = Field(..., description="나이", ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "pregnancies": 2,
                "glucose": 130.0,
                "blood_pressure": 70.0,
                "skin_thickness": 20.0,
                "insulin": 80.0,
                "bmi": 25.5,
                "diabetes_pedigree": 0.45,
                "age": 35
            }
        }
    }

class ChatInput(BaseModel):
    prediction_result: dict
    user_message: str

# 4. 프론트엔드 서빙 및 상태 엔드포인트
@app.get("/")
def serve_ui():
    """메인 페이지(index.html)를 반환합니다."""
    return FileResponse("index.html")

@app.get("/api/status")
def read_status():
    """서버 상태 및 토큰 사용량을 반환합니다."""
    return {
        "status": "online",
        "token_usage": TOKEN_USAGE_TRACKER
    }

# 5. 당뇨병 위험도 예측 엔드포인트
@app.post("/predict", summary="당뇨병 위험도 예측 API")
def predict_diabetes(data: DiabetesInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="머신러닝 모델이 준비되지 않았습니다.")
    
    try:
        input_features = np.array([[
            data.pregnancies,
            data.glucose,
            data.blood_pressure,
            data.skin_thickness,
            data.insulin,
            data.bmi,
            data.diabetes_pedigree,
            data.age
        ]])
        
        input_scaled = scaler.transform(input_features)
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        diabetes_probability = round(probabilities[1] * 100, 2)
        
        result = {
            "is_diabetes": int(prediction),
            "diabetes_risk_percent": diabetes_probability,
            "risk_level": "고위험군" if diabetes_probability >= 50 else "안정/주의군",
            "input_data": data.model_dump()
        }
        
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 추론 중 에러 발생: {str(e)}")

# 6. AI 상담 엔드포인트 (OpenAI 통합 및 비용 방어 로직)
@app.post("/chat", summary="AI 건강 상담 API")
def chat_counseling(data: ChatInput):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OpenAI API 키가 설정되지 않았습니다.")

    # 1) 시스템 프롬프트 구성
    system_prompt = (
        "당신은 전문적인 당뇨병 건강 상담가입니다. "
        "사용자의 당뇨병 예측 결과와 입력 데이터를 바탕으로 친절하고 전문적인 조언을 제공하세요. "
        "의학적 진단은 아니며 참고용임을 명시하세요."
    )
    
    # 2) 사용자 컨텍스트 구성
    context = f"예측 결과: {data.prediction_result}\n사용자 질문: {data.user_message}"
    
    # 3) 토큰 계산 및 비용 방어
    total_input_tokens = count_tokens(system_prompt + context)
    if total_input_tokens > MAX_TOKEN_LIMIT:
        raise HTTPException(status_code=400, detail="입력 데이터가 너무 큽니다 (토큰 제한 초과).")

    try:
        # 4) OpenAI API 호출
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context}
            ],
            max_tokens=500
        )
        
        ai_message = response.choices[0].message.content
        used_tokens = response.usage.total_tokens
        
        # 5) 사용량 트래킹
        TOKEN_USAGE_TRACKER["total_tokens"] += used_tokens
        AI_TOKEN_USAGE_COUNTER.inc(used_tokens)  # Prometheus 커스텀 메트릭 증가
        
        return {
            "ai_response": ai_message,
            "tokens_used": used_tokens,
            "cumulative_tokens": TOKEN_USAGE_TRACKER["total_tokens"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI 상담 중 에러 발생: {str(e)}")