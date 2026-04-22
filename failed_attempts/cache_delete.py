import os
import glob

def clear_yolo_cache():
    # 데이터셋 라벨이 있는 최상위 폴더 경로 (r을 붙여 경로 인식 오류 방지)
    base_labels_dir = r"E:\Robot_Team_Project\Python_Files(AI Programming)\TestSet02\dataset\labels"
    
    print("🧹 YOLO 캐시 파일 정리를 시작합니다...")
    
    # 해당 폴더와 하위 폴더(train, val, test)에 있는 모든 .cache 파일 찾기
    search_pattern = os.path.join(base_labels_dir, '**', '*.cache')
    cache_files = glob.glob(search_pattern, recursive=True)
    
    if not cache_files:
        print("✨ 삭제할 캐시 파일이 없습니다. 이미 깨끗한 상태입니다!")
        return

    # 찾은 캐시 파일들 모두 삭제
    for file_path in cache_files:
        try:
            os.remove(file_path)
            print(f"✅ 삭제 완료: {file_path}")
        except Exception as e:
            print(f"❌ 삭제 실패 ({file_path}): {e}")
            
    print("🚀 모든 캐시가 삭제되었습니다. 이제 완벽하게 새로운 라벨로 학습할 준비가 되었습니다!")

if __name__ == "__main__":
    clear_yolo_cache()