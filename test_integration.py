import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    print("1. 헬스체크 테스트 중...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"상태 코드: {response.status_code}")
        print(f"응답: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"오류: {e}")
        return False

def test_predict():
    print("\n2. 당뇨병 예측 API 테스트 중...")
    data = {
        "pregnancies": 2,
        "glucose": 130.0,
        "blood_pressure": 70.0,
        "skin_thickness": 20.0,
        "insulin": 80.0,
        "bmi": 25.5,
        "diabetes_pedigree": 0.45,
        "age": 35
    }
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"상태 코드: {response.status_code}")
    print(f"결과: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    return response.status_code == 200, response.json()

def test_chat(prediction_result):
    print("\n3. AI 상담 API 테스트 중 (비용 방어 로직 확인)...")
    data = {
        "prediction_result": prediction_result,
        "user_message": "내 결과가 어떤 의미인지 설명해주고, 식단 조언을 해줘."
    }
    response = requests.post(f"{BASE_URL}/chat", json=data)
    print(f"상태 코드: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"AI 응답 요약: {result['ai_response'][:100]}...")
        print(f"사용된 토큰: {result['tokens_used']}")
        print(f"누적 토큰: {result['cumulative_tokens']}")
        return True
    else:
        print(f"실패 응답: {response.text}")
        return False

def test_metrics():
    print("\n4. Prometheus 메트릭 엔드포인트 테스트 중...")
    response = requests.get(f"{BASE_URL}/metrics")
    print(f"상태 코드: {response.status_code}")
    if response.status_code == 200:
        if "ai_counseling_tokens_total" in response.text:
            print("✅ 커스텀 메트릭(ai_counseling_tokens_total)이 정상적으로 노출되고 있습니다.")
        else:
            print("⚠️ 커스텀 메트릭을 찾을 수 없습니다.")
        return True
    return False

if __name__ == "__main__":
    print("=== 시스템 통합 테스트 시작 ===")
    if test_health():
        success, prediction = test_predict()
        if success:
            test_chat(prediction)
            test_metrics()
    else:
        print("❌ 서버가 실행 중이지 않습니다. 'uvicorn main:app --reload'를 실행해 주세요.")
    print("\n=== 테스트 종료 ===")
