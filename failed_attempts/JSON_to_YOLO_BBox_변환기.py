import json
import os
from pathlib import Path

# 1. 데이터셋 경로 설정 (data.yaml에 적힌 경로 기준)
# 만약 현재 C 드라이브(OneDrive)에서 작업 중이시라면 이 경로를 수정해 주세요.
BASE_DIR = Path(r"E:\Robot_Team_Project\Python_Files(AI Programming)\TestSet02\dataset")

# 2. 클래스 맵핑 (data.yaml의 names 순서와 반드시 동일해야 합니다)
CLASS_MAP = {
    'inspection_C4_Bomb': 0,
    'inspection_Communication': 1,
    'inspection_Supply': 2,
}

def convert_json_to_yolo_bbox(json_path: Path, out_dir: Path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    W = data['imageWidth']
    H = data['imageHeight']
    lines = []

    for shape in data['shapes']:
        label = shape['label']
        if label not in CLASS_MAP:
            continue
            
        cls_id = CLASS_MAP[label]
        pts = shape['points']

        # [핵심 수학 계산]
        # 다각형(Polygon)이든 사각형(Rectangle)이든 상관없이,
        # 찍힌 모든 점들 중에서 가장 왼쪽/오른쪽(x), 가장 위/아래(y) 좌표를 찾습니다.
        x_coords = [p[0] for p in pts]
        y_coords = [p[1] for p in pts]
        
        xmin = min(x_coords)
        xmax = max(x_coords)
        ymin = min(y_coords)
        ymax = max(y_coords)

        # YOLO BBox 규칙: 중심점 좌표, 너비, 높이 구하기
        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        box_w = xmax - xmin
        box_h = ymax - ymin

        # 정규화(Normalization): 0 ~ 1 사이의 비율로 만들기 위해 전체 이미지 크기로 나눔
        x_center /= W
        y_center /= H
        box_w /= W
        box_h /= H

        # 소수점 6자리까지 포맷팅하여 한 줄 완성
        line = f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
        lines.append(line)

    if lines:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (json_path.stem + ".txt")
        out_path.write_text("\n".join(lines), encoding="utf-8")

def main():
    print("=" * 60)
    print(f"현재 설정된 최상위 경로: {BASE_DIR}")
    if not BASE_DIR.exists():
        print(">>> [경고] 위 경로의 폴더가 존재하지 않습니다!")
        print(">>> E 드라이브가 맞는지, 혹은 OneDrive 경로인지 다시 확인해 주세요.")
        print("=" * 60)
        return
    else:
        print(">>> [성공] 경로를 찾았습니다. 변환을 준비합니다.")
    print("=" * 60)

    total_converted = 0

    # train, val, test 폴더 내부를 모두 순회하며 변환 실행
    for split in ['train', 'val', 'test']:
        labels_dir = BASE_DIR / 'labels' / split
        images_dir = BASE_DIR / 'images' / split
        out_dir = BASE_DIR / 'labels' / split  # 생성된 txt를 저장할 최종 폴더
        
        # 1. 먼저 labels 폴더에서 json 파일을 찾습니다.
        json_files = list(labels_dir.glob("*.json"))
        search_path = labels_dir
        
        # 2. labels 폴더에 없다면, images 폴더를 확인합니다 (LabelMe 특성)
        if not json_files and images_dir.exists():
            json_files = list(images_dir.glob("*.json"))
            if json_files:
                search_path = images_dir
                
        print(f"\n[{split.upper()} 폴더 확인]")
        print(f" - 탐색한 경로: {search_path}")
        print(f" - 찾은 JSON 파일 수: {len(json_files)}개")

        if len(json_files) > 0:
            for jf in json_files:
                convert_json_to_yolo_bbox(jf, out_dir)
            total_converted += len(json_files)
            print(f" -> [완료] {out_dir} 폴더에 txt 파일 저장됨")

    print("\n" + "=" * 60)
    print(f"최종 결과: 총 {total_converted}개의 파일이 성공적으로 변환되었습니다.")
    print("=" * 60)

if __name__ == "__main__":
    main()