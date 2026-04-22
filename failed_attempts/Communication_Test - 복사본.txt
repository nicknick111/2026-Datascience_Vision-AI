import pymysql
import requests

# ==========================================
# 1. MySQL & API 서버 설정
# ==========================================
DB_CONFIG = {
    "host": "192.168.3.21",
    "database": "dronedb",
    "user": "root",
    "password": "1234",
    "port": 3306
}

# 연동할 API 서버 주소 (실제 운영하시는 API 서버 주소로 변경하세요)
API_ENDPOINT = "http://192.168.3.21:8080/api/training-data"

def check_system_status():
    """
    DB 및 API 서버의 접속 상태를 한눈에 파악하기 위한 전용 점검 함수입니다.
    """
    print("\n" + "=" * 50)
    print("[시스템 점검] 서버 연결 상태를 확인합니다...")
    print("=" * 50)
    
    # 1. DB 접속 상태 점검
    try:
        # 짧은 타임아웃(3초)을 주어 연결 시도
        conn = pymysql.connect(**DB_CONFIG, connect_timeout=3)
        print("  🟢 [DB 접속 상태] 정상: MySQL 서버 통신 가능")
        conn.close()
    except Exception as e:
        print(f"  🔴 [DB 접속 상태] 실패: MySQL 서버를 찾을 수 없거나 권한이 없습니다.")
        print(f"      -> 상세 원인: {e}")

    # 2. API 서버 접속 상태 점검
    try:
        # API 서버가 살아있는지 GET 요청으로 가볍게 찔러봅니다 (타임아웃 3초)
        res = requests.get(API_ENDPOINT, timeout=3)
        print(f"  🟢 [API 접속 상태] 정상: API 서버 통신 가능 (응답 코드: {res.status_code})")
    except requests.exceptions.RequestException as e:
        print(f"  🔴 [API 접속 상태] 실패: API 서버에 접속할 수 없거나 서버가 꺼져있습니다.")
        print(f"      -> 상세 원인: {e}")
        
    print("=" * 50 + "\n")

# ==========================================
# 메인 실행부
# ==========================================
if __name__ == "__main__":
    check_system_status()
    print("엔터(Enter) 키를 누르면 종료됩니다...")
    input()