import cv2
import mediapipe as mp
import pygame
import os
import numpy as np
import math

# pygame 초기화
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

# 바이올린 샘플 파일 경로
SAMPLE_DIR = '/Users/minseung/Desktop/noinstband/data/sounds/violin'

# 손 모양별 음계 매핑
GESTURE_NOTES = {
    'fist': 'violinC4',           # 주먹
    'one': 'violinD4',            # 검지 1개
    'two': 'violinE4',            # 검지+중지
    'three': 'violinF4',          # 3개
    'four': 'violinG4',           # 4개
    'thumb': 'violinA4',          # 엄지만
    'thumb_one': 'violinB4',      # 엄지+검지
    'all': 'violinC5'             # 모두
}

# 피타고라스 음률 (순정율 주파수 비율)
# 기준음 C4 = 1/1
PYTHAGOREAN_RATIOS = {
    'fist': (1, 1),           # C (도) - 1/1 = 1.0
    'one': (9, 8),            # D (레) - 9/8 = 1.125
    'two': (81, 64),          # E (미) - 81/64 = 1.265625
    'three': (4, 3),          # F (파) - 4/3 = 1.333...
    'four': (3, 2),           # G (솔) - 3/2 = 1.5
    'thumb': (27, 16),        # A (라) - 27/16 = 1.6875
    'thumb_one': (243, 128),  # B (시) - 243/128 = 1.8984375
    'all': (2, 1)             # C5 (높은 도) - 2/1 = 2.0 (옥타브)
}

# 음계별 색상 (RGB)
NOTE_COLORS = {
    'fist': (255, 0, 0),         # 빨강
    'one': (255, 127, 0),        # 주황
    'two': (255, 255, 0),        # 노랑
    'three': (0, 255, 0),        # 초록
    'four': (0, 127, 255),       # 파랑
    'thumb': (0, 0, 255),        # 남색
    'thumb_one': (127, 0, 255),  # 보라
    'all': (255, 0, 127)         # 자홍
}

# 사운드 로드
sounds = {}
print("=" * 60)
print("🎻 피타고라스 프랙탈 오케스트라")
print("=" * 60)
print("바이올린 샘플 로딩 중...")

for gesture, filename in GESTURE_NOTES.items():
    filepath = os.path.join(SAMPLE_DIR, f"{filename}.mp3")
    
    if os.path.exists(filepath):
        sounds[gesture] = pygame.mixer.Sound(filepath)
        ratio = PYTHAGOREAN_RATIOS[gesture]
        print(f"✅ {gesture}: {filename}.mp3 (비율: {ratio[0]}/{ratio[1]} = {ratio[0]/ratio[1]:.4f})")
    else:
        print(f"⚠️ 파일 없음: {filepath}")

if not sounds:
    print("\n❌ 재생할 샘플 파일이 없습니다!")
    print(f"📁 경로 확인: {SAMPLE_DIR}")
    exit()

print(f"\n✅ 총 {len(sounds)}개 샘플 로드 완료!")
print("=" * 60)

# 피타고라스 프랙탈 파티클 클래스
class PythagoreanParticle:
    def __init__(self, x, y, color, ratio_tuple):
        self.x = x
        self.y = y
        self.color = color
        self.life = 1.0
        self.angle = np.random.uniform(0, 2 * math.pi)
        
        # 피타고라스 비율 계산
        numerator, denominator = ratio_tuple
        self.ratio = numerator / denominator
        
        # 비율에 따른 프랙탈 차원 결정
        # 1.0 (완전1도) ~ 2.0 (옥타브) 사이를 1~8 가지로 매핑
        self.branches = int(3 + (self.ratio - 1.0) * 5)  # 3~8 가지
        self.branches = max(3, min(12, self.branches))
        
        # 재귀 깊이 (협화도에 따라)
        # 완전협화음(1/1, 3/2, 2/1)은 깊게, 불협화음은 얕게
        if denominator in [1, 2, 3, 4]:
            self.recursion_depth = 3
        else:
            self.recursion_depth = 2
        
        # 크기는 비율에 비례
        self.base_size = 20 + (self.ratio * 30)
        
        # 회전 속도도 비율 반영
        self.rotation_speed = (self.ratio - 1.0) * 0.15
        
    def update(self):
        self.life -= 0.01
        self.angle += self.rotation_speed
        return self.life > 0
    
    def draw_recursive_fractal(self, frame, x, y, length, angle, depth, alpha_mult=1.0):
        """재귀적으로 프랙탈 가지 그리기"""
        if depth <= 0 or length < 2:
            return
        
        # 현재 가지의 끝점 계산
        end_x = int(x + math.cos(angle) * length)
        end_y = int(y + math.sin(angle) * length)
        
        # 투명도 계산
        alpha = self.life * alpha_mult * (depth / self.recursion_depth)
        color = tuple(int(c * alpha) for c in self.color)
        
        # 선 두께는 깊이에 따라 감소
        thickness = max(1, int(3 * (depth / self.recursion_depth)))
        
        # 가지 그리기
        cv2.line(frame, (int(x), int(y)), (end_x, end_y), color, thickness)
        
        # 끝에 작은 원 (에너지 포인트)
        if depth == self.recursion_depth:
            cv2.circle(frame, (end_x, end_y), int(thickness * 1.5), color, -1)
        
        # 재귀: 여러 갈래로 분기
        # 분기 각도는 피타고라스 비율에 기반
        angle_spread = (2 * math.pi) / self.branches
        
        for i in range(self.branches):
            new_angle = angle + angle_spread * i
            # 길이는 황금비율에 가깝게 감소 (0.618)
            new_length = length * 0.618 * self.ratio / 1.5
            
            self.draw_recursive_fractal(
                frame, end_x, end_y, new_length, 
                new_angle, depth - 1, alpha_mult * 0.7
            )
    
    def draw(self, frame):
        if self.life <= 0:
            return
        
        # 중심에서 재귀적 프랙탈 그리기
        initial_length = self.base_size * self.life
        
        self.draw_recursive_fractal(
            frame, self.x, self.y, 
            initial_length, self.angle, 
            self.recursion_depth
        )
        
        # 중앙 코어 (음의 근원)
        core_size = int(5 * self.life * self.ratio / 1.5)
        glow_size = int(core_size * 2)
        
        # 외곽 글로우
        cv2.circle(frame, (int(self.x), int(self.y)), glow_size, 
                  tuple(int(c * 0.3 * self.life) for c in self.color), -1)
        # 내부 코어
        cv2.circle(frame, (int(self.x), int(self.y)), core_size, self.color, -1)
        
        # 비율 표시 (디버깅용 - 원하면 주석 처리)
        # cv2.putText(frame, f"{self.ratio:.3f}", 
        #            (int(self.x) + 10, int(self.y) - 10),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.color, 1)

# 파티클 리스트
particles = []

def play_sound(gesture):
    """특정 손 모양의 소리 재생"""
    if gesture in sounds:
        sounds[gesture].play()

# Mediapipe 손 인식 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# 웹캠 열기
cap = cv2.VideoCapture(0)

def fingers_up(hand_landmarks):
    """어떤 손가락이 펴져있는지 확인"""
    fingers = []
    
    # 엄지 (왼손/오른손 구분)
    if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x > \
       hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
        fingers.append(True)
    else:
        fingers.append(False)
    
    # 나머지 손가락 (검지, 중지, 약지, 새끼)
    for id in [mp_hands.HandLandmark.INDEX_FINGER_TIP,
               mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
               mp_hands.HandLandmark.RING_FINGER_TIP,
               mp_hands.HandLandmark.PINKY_TIP]:
        tip_y = hand_landmarks.landmark[id].y
        pip_y = hand_landmarks.landmark[id - 2].y
        fingers.append(tip_y < pip_y)
    
    return fingers

# 이전 손 모양 저장
previous_gesture = None

print("\n🎻 피타고라스 프랙탈 오케스트라 시작!")
print("\n✋ 손가락 패턴 (피타고라스 음률):")
print("  주먹 = 도 (C4) - 1/1")
print("  검지 = 레 (D4) - 9/8") 
print("  검지+중지 = 미 (E4) - 81/64")
print("  검지+중지+약지 = 파 (F4) - 4/3")
print("  검지+중지+약지+새끼 = 솔 (G4) - 3/2")
print("  엄지만 = 라 (A4) - 27/16")
print("  엄지+검지 = 시 (B4) - 243/128")
print("  모두 펴기 = 높은 도 (C5) - 2/1")
print("\n✨ 각 음정의 주파수 비율이 프랙탈 차원으로 표현됩니다!")
print("🛑 종료: ESC 키")
print("=" * 60)

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 웹캠을 열 수 없습니다.")
        break
    
    # 좌우 반전
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    # BGR을 RGB로 변환
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 손 인식
    result = hands.process(rgb)
    
    # 파티클 업데이트 및 그리기
    particles = [p for p in particles if p.update()]
    for particle in particles:
        particle.draw(frame)
    
    # 제목
    cv2.putText(frame, "Pythagorean Fractal Orchestra - ESC to quit", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # 손이 감지되면
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # 손 그리기
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            fingers = fingers_up(hand_landmarks)
            
            current_gesture = None
            gesture_name = ""
            note_name = ""
            ratio_text = ""
            
            if fingers == [False, False, False, False, False]:
                current_gesture = 'fist'
                gesture_name = "Fist"
                note_name = "Do (C4)"
                ratio_text = "1/1"
                
            elif fingers == [False, True, False, False, False]:
                current_gesture = 'one'
                gesture_name = "One"
                note_name = "Re (D4)"
                ratio_text = "9/8"
                
            elif fingers == [False, True, True, False, False]:
                current_gesture = 'two'
                gesture_name = "Two"
                note_name = "Mi (E4)"
                ratio_text = "81/64"
                
            elif fingers == [False, True, True, True, False]:
                current_gesture = 'three'
                gesture_name = "Three"
                note_name = "Fa (F4)"
                ratio_text = "4/3"
                
            elif fingers == [False, True, True, True, True]:
                current_gesture = 'four'
                gesture_name = "Four"
                note_name = "Sol (G4)"
                ratio_text = "3/2"
                
            elif fingers == [True, False, False, False, False]:
                current_gesture = 'thumb'
                gesture_name = "Thumb"
                note_name = "La (A4)"
                ratio_text = "27/16"
                
            elif fingers == [True, True, False, False, False]:
                current_gesture = 'thumb_one'
                gesture_name = "Thumb+One"
                note_name = "Si (B4)"
                ratio_text = "243/128"
                
            elif fingers == [True, True, True, True, True]:
                current_gesture = 'all'
                gesture_name = "All"
                note_name = "High Do (C5)"
                ratio_text = "2/1"
            
            # 손 모양이 바뀌었을 때 재생 + 프랙탈 생성
            if current_gesture and current_gesture != previous_gesture:
                play_sound(current_gesture)
                previous_gesture = current_gesture
                
                # 손끝에서 피타고라스 프랙탈 생성
                fingertip_indices = [4, 8, 12, 16, 20]
                color = NOTE_COLORS.get(current_gesture, (255, 255, 255))
                ratio = PYTHAGOREAN_RATIOS[current_gesture]
                
                for tip_idx in fingertip_indices:
                    landmark = hand_landmarks.landmark[tip_idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    
                    # 각 손끝마다 2개의 프랙탈 파티클
                    for _ in range(2):
                        particles.append(PythagoreanParticle(x, y, color, ratio))
            
            # 화면에 정보 표시
            if note_name:
                cv2.putText(frame, note_name, 
                           (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 150, 0), 3)
                cv2.putText(frame, f"Ratio: {ratio_text}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 255), 2)
                cv2.putText(frame, f"({gesture_name})", 
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
    else:
        cv2.putText(frame, "Show your hand!", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    
    # 화면 표시
    cv2.imshow("Pythagorean Fractal Orchestra", frame)
    
    # ESC 키로 종료
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# 정리
cap.release()
cv2.destroyAllWindows()
pygame.quit()
print("\n👋 연주 종료! 수고하셨습니다!")
