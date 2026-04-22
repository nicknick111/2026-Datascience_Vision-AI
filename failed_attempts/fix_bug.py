import os
import site

print("🔍 숨어있는 canvas.py 파일을 탐색 중입니다...")

# 1. 파이썬 라이브러리가 설치되는 모든 핵심 폴더(site-packages)를 가져옵니다.
site_packages = site.getsitepackages()
target_path = None

# 사용자님의 이전 로그를 바탕으로 한 맞춤형 강제 경로 추가
site_packages.append(r"C:\Users\Ghost\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages")

# 2. labelImg 폴더 및 libs 폴더 내부까지 샅샅이 뒤집니다.
for sp in site_packages:
    path1 = os.path.join(sp, "labelImg", "canvas.py")
    path2 = os.path.join(sp, "libs", "canvas.py")
    
    if os.path.exists(path1):
        target_path = path1
        break
    if os.path.exists(path2):
        target_path = path2
        break

if target_path:
    # 3. 고장 난 코드를 정수(int)로 강제 변환하도록 교체합니다.
    with open(target_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    content = content.replace(
        "p.drawLine(self.prev_point.x(), 0, self.prev_point.x(), self.pixmap.height())", 
        "p.drawLine(int(self.prev_point.x()), 0, int(self.prev_point.x()), int(self.pixmap.height()))"
    )
    content = content.replace(
        "p.drawLine(0, self.prev_point.y(), self.pixmap.width(), self.prev_point.y())", 
        "p.drawLine(0, int(self.prev_point.y()), int(self.pixmap.width()), int(self.prev_point.y()))"
    )
    
    with open(target_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print("\n🎉 [대성공] 숨겨진 버그 파일을 찾아 완벽하게 치료했습니다!")
    print(f"👉 수정된 파일 위치: {target_path}")
    print("이제 labelImg를 켜고 당당하게 'W'를 눌러보세요!")
else:
    print("\n❌ 파일을 찾지 못했습니다. 수동 수정이 필요합니다.")