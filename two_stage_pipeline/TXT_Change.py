import os

def fix_mixed_yolo_formats(labels_dirs):
    total_fixed = 0
    total_ok = 0
    
    print("🚀 전체 데이터셋 라벨 포맷(OBB) 일괄 검사 및 복구를 시작합니다...\n")

    for labels_dir in labels_dirs:
        # 경로가 존재하는지 확인 (test 폴더가 없을 수도 있으므로 예외 처리)
        if not os.path.exists(labels_dir):
            print(f"⚠️ 폴더를 찾을 수 없어 건너뜁니다: {labels_dir}")
            continue
            
        print(f"📂 검사 중: {labels_dir}")
        dir_fixed = 0
        dir_ok = 0
        
        for filename in os.listdir(labels_dir):
            if not filename.endswith(".txt") or filename == "classes.txt":
                continue
                
            filepath = os.path.join(labels_dir, filename)
            
            with open(filepath, 'r') as f:
                lines = f.readlines()
                
            new_lines = []
            needs_fix = False
            
            for line in lines:
                parts = line.strip().split()
                
                # 1. 정상 OBB (클래스 1 + 좌표 8)
                if len(parts) == 9:
                    new_lines.append(line)
                
                # 2. 복구가 필요한 일반 바운딩 박스 (클래스 1 + cx, cy, w, h)
                elif len(parts) == 5:
                    needs_fix = True
                    cls_id = parts[0]
                    cx, cy, w, h = map(float, parts[1:5])
                    
                    # 수평 박스를 0도 기준 4점 폴리곤 좌표로 변환
                    x1, y1 = cx - w/2, cy - h/2  # 좌상단
                    x2, y2 = cx + w/2, cy - h/2  # 우상단
                    x3, y3 = cx + w/2, cy + h/2  # 우하단
                    x4, y4 = cx - w/2, cy + h/2  # 좌하단
                    
                    obb_line = f"{cls_id} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}\n"
                    new_lines.append(obb_line)
                else:
                    new_lines.append(line)
                    
            # 파일이 수정되었다면 덮어쓰기 저장
            if needs_fix:
                with open(filepath, 'w') as f:
                    f.writelines(new_lines)
                dir_fixed += 1
            else:
                dir_ok += 1
        
        print(f"   👉 결과: 정상 OBB 유지 {dir_ok}개 / 일반 박스 복구 {dir_fixed}개\n")
        total_fixed += dir_fixed
        total_ok += dir_ok

    print("=" * 50)
    print(f"✅ 전체 작업 완료! (총 정상: {total_ok}개 / 총 복구: {total_fixed}개)")
    print("이제 완벽한 OBB 포맷으로 통일되었습니다. 최종 학습을 진행하셔도 좋습니다!")

# 장원영 님의 프로젝트 전체 라벨 폴더 경로 리스트
label_paths = [
    r"E:\Robot_Team_Project\Python_Files(AI Programming)\TestSet02\Dataset_OBB\train\labels",
    r"E:\Robot_Team_Project\Python_Files(AI Programming)\TestSet02\Dataset_OBB\val\labels",
    r"E:\Robot_Team_Project\Python_Files(AI Programming)\TestSet02\Dataset_OBB\test\labels"
]

if __name__ == '__main__':
    fix_mixed_yolo_formats(label_paths)