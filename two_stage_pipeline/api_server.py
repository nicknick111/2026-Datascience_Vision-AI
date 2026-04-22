# api_server.py
import asyncio            # ✨ [V1 대비 핵심 개선점] 비동기 처리 라이브러리
import time
import threading
import datetime
import requests
import cv2
from queue import Queue
from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn
from typing import Dict, Any 

import vision_ai_main

API_ENDPOINT = "http://192.168.3.62:8000/inspection"
app = FastAPI()

# ==========================================
# [데이터 모델 정의]
# ==========================================
class OrderRequest(BaseModel):
    target_product: str

class ApiUploader:
    """[V1 동일] HTTP 전송 시 딜레이가 비전 AI에 영향을 주지 않도록 분리된 업로드 일꾼"""
    def __init__(self, url):
        self.url = url
        self.queue = Queue(maxsize=30)
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            payload = self.queue.get()
            try: requests.post(self.url, json=payload, timeout=1.0)
            except: pass
            self.queue.task_done()

    def send(self, data):
        if not self.queue.full(): self.queue.put(data)

uploader = ApiUploader(API_ENDPOINT)

def handle_ai_decision(result_status, part_name):
    """비전 엔진이 판정을 끝내면 외부 메인 서버로 데이터를 발송합니다."""
    payload = {
        "product_id": vision_ai_main.current_order_id, 
        "result": result_status,
        "timestamp": datetime.datetime.now().isoformat(),
        "camera_id": "RPC-20F"
    }
    uploader.send(payload)

# ==========================================
# [API 엔드포인트 라우팅]
# ==========================================

@app.get("/")
@app.get("/monitor", response_class=HTMLResponse)
def view_monitor():
    """
    ✨ [V1에 없던 기능] 웹 기반 실시간 모니터링 대시보드
    공장 관리자가 http://서버IP:8001/monitor 로 접속하면 볼 수 있는 HTML 페이지를 내려줍니다.
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Vision AI 실시간 모니터링</title>
        <style>
            body { background-color: #121212; color: #e0e0e0; font-family: sans-serif; text-align: center; }
            .video-container { display: inline-block; background-color: #1e1e1e; padding: 15px; border-radius: 12px; }
            img { max-width: 100%; border-radius: 8px; border: 1px solid #333; }
        </style>
    </head>
    <body>
        <h2>🎯 통합 Vision AI 품질 검사 라인</h2>
        <div class="video-container">
            <img src="/video_feed" alt="실시간 카메라 스트리밍 연결 중...">
        </div>
    </body>
    </html>
    """
    return html_content

def health_check():
    """PLC에서 서버가 살아있는지 주기적으로 확인할 때 씁니다."""
    return {"status": "OK", "message": "AI Vision Server is Running."}

@app.post("/trigger")
def trigger_vision(body: dict = Body(default={})): 
    """
    ✨ [V1 대비 획기적 개선] Non-Blocking(비동기 대기) 트리거
    V1에서는 여기서 while 문을 돌며 AI가 판정할 때까지 무식하게 10초씩 기다리다 보니 
    서버가 먹통(Time-out)되는 현상이 있었습니다.
    현재 버전은 '비전 엔진에 깃발만 꽂아두고 즉시 OK 응답을 반환'하여 멈춤을 방지합니다.
    """
    vision_ai_main.current_order_id = body.get("order_id", f"UNKNOWN-{int(time.time())}")
    vision_ai_main.trigger_flag = True
    vision_ai_main.current_status = "WAITING" 
    
    print(f"[API] 촬영 신호 수신 완료 (Order ID: {vision_ai_main.current_order_id})")
    return {"message": "촬영 신호 수신 완료"} # 즉시 응답

@app.get("/result")
def get_result():
    """
    ✨ [V1의 문제 해결사] Polling 방식의 결과 조회
    위 /trigger에서 깃발만 꽂고 리턴했으므로, PLC나 메인 서버는 이 API를 0.5초마다 찔러서
    status가 "PASS" 또는 "FAIL"로 바뀌었는지 확인(Polling)하여 결과를 가져갑니다.
    """
    conf_ratio = round(vision_ai_main.current_confidence / 100.0, 3)
    
    return {
        "status": vision_ai_main.current_status,         
        "confidence": conf_ratio,                        
        "part_name": vision_ai_main.current_part_name,   
        "order_id": vision_ai_main.current_order_id,     
        "timestamp": datetime.datetime.now().isoformat()
    }

# ==========================================
# [수정된 비동기 영상 스트리밍 코드]
# ==========================================
@app.get('/video_feed')
async def video_feed(): 
    """
    ✨ [V1 대비 개선] async/await 비동기 스트리밍 적용
    V1에서는 영상 인코딩 도중 다른 API(트리거 등)가 들어오면 처리가 막혔습니다.
    async 구조로 변경하여 다른 API 요청이 영상 스트리밍과 완벽히 독립적으로 처리됩니다.
    """
    async def generate():
        while vision_ai_main.is_running:
            current_frame = None
            
            # 자물쇠(Lock)를 잡는 시간을 최소화하기 위해 프레임 복사만 빠르게 수행
            with vision_ai_main.lock:
                if vision_ai_main.global_frame is not None:
                    current_frame = vision_ai_main.global_frame.copy()
            
            if current_frame is not None:
                ret, jpeg = cv2.imencode('.jpg', current_frame)
                if ret:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            
            # 다른 웹/API 요청이 끼어들 수 있도록 숨을 쉬어주는(Yield) 역할
            await asyncio.sleep(0.05) 
            
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/set_order")
def set_order(order: OrderRequest): 
    vision_ai_main.expected_product = order.target_product
    msg = f"[API] 작업 지시 수신: '{order.target_product}'"
    print(msg)
    vision_ai_main.system_log(msg)
    return {"message": "정상적으로 변경됨", "current_target": order.target_product}

@app.get("/get_logs")
def get_logs():
    logs = []
    while not vision_ai_main.log_queue.empty(): logs.append(vision_ai_main.log_queue.get())
    return {"logs": logs}

# ==========================================
# [메인 실행 블록]
# ==========================================
if __name__ == '__main__':
    vision_thread = threading.Thread(
        target=vision_ai_main.run_pipeline, 
        kwargs={"cam_index": 0, "on_decision_callback": handle_ai_decision}
    )
    vision_thread.daemon = True
    vision_thread.start()
    
    # 외부 시스템과의 포트 충돌을 막기 위해 8001번 포트 사용
    print("🚀 API 통신 메인 서버 가동 (0.0.0.0:8001)")
    uvicorn.run(app, host="0.0.0.0", port=8001)