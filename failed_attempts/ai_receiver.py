from fastapi import FastAPI
import cv2
import datetime
import requests
from requests.exceptions import RequestException

app = FastAPI()

# 대상 API 서버 주소
API_SERVER = "http://192.168.3.21:8000"

@app.post("/trigger")
def trigger_vision():
    """API 서버에서 촬영 신호 받으면 카메라 작동"""
    print("[AI] 촬영 신호 수신! 카메라 작동 중...")

    # 임시 판정 결과 (나중에 YOLOv8n으로 교체)
    result = "정상"
    print(f"[AI] 판정 결과: {result}")

    # 판정 결과 API 서버로 전송할 데이터 세팅
    data = {
        "product_id": f"DRONE-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
        "result": result,
        "timestamp": datetime.datetime.now().isoformat(),
        "camera_id": "RPC-20F"
    }
    
    # ---------------------------------------------------------
    # 수정한 부분: 안전하게 서버로 전송을 시도하는 try-except 블록
    # ---------------------------------------------------------
    try:
        # timeout=5 를 주어 5초 이상 응답이 없으면 포기하게 만듭니다.
        response = requests.post(f"{API_SERVER}/inspection", json=data, timeout=5)
        print(f"[API] 전송 완료 → 상태코드: {response.status_code}")
        
    except RequestException as e:
        # 서버 접속에 실패하더라도 프로그램이 죽지 않고 에러 메시지만 출력합니다.
        print(f"[API 에러] 서버와 접속할 수 없습니다: {e}")
        # 실패했을 때의 결과를 반환하고 함수를 종료합니다.
        return {"message": "판정 완료되었으나 서버 전송 실패", "result": result}

    # 성공했을 때의 결과를 반환합니다.
    return {"message": "판정 완료", "result": result}