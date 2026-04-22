import psycopg2

# 본인의 정보로 수정하세요
DB_CONFIG = {
    "host": "localhost",
    "database": "postgres", # 기본 DB 이름
    "user": "postgres",     # 기본 사용자명
    "password": "1234" 
}

def test_connection():
    try:
        print("[TEST] DB에 접속을 시도합니다...")
        conn = psycopg2.connect(**DB_CONFIG)
        print("[SUCCESS] DB 연결에 성공했습니다!")
        
        cur = conn.cursor()
        cur.execute("SELECT version();")
        db_version = cur.fetchone()
        print(f"[INFO] PostgreSQL 버전: {db_version}")
        
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[FAIL] 연결 실패 사유: {e}")

if __name__ == "__main__":
    test_connection()