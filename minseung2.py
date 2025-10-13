import cv2
import mediapipe as mp
import pygame
import os

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

# 사운드 로드
sounds = {}
print("=" * 50)
print("🎻 바이올린 샘플 로딩 중...")
print("=" * 50)

for gesture, filename in GESTURE_NOTES.items():
    # .wav 확장자 추가해서 경로 만들기
    filepath = os.path.join(SAMPLE_DIR, f"{filename}.mp3")
    
    if os.path.exists(filepath):
        sounds[gesture] = pygame.mixer.Sound(filepath)
        print(f"✅ {gesture}: {filename}.mp3")
    else:
        print(f"⚠️ 파일 없음: {filepath}")

if not sounds:
    print("\n❌ 재생할 샘플 파일이 없습니다!")
    print(f"📁 경로 확인: {SAMPLE_DIR}")
    exit()

print(f"\n✅ 총 {len(sounds)}개 샘플 로드 완료!")
print("=" * 50)

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

# 이전 손 모양 저장 (같은 동작 반복 방지)
previous_gesture = None

print("\n🎻 바이올린 손짓 오케스트라 시작!")
print("✋ 손가락 패턴:")
print("  주먹 = 도 (C4)")
print("  검지 = 레 (D4)")
print("  검지+중지 = 미 (E4)")
print("  검지+중지+약지 = 파 (F4)")
print("  검지+중지+약지+새끼 = 솔 (G4)")
print("  엄지만 = 라 (A5)")
print("  엄지+검지 = 시 (B5)")
print("  모두 펴기 = 높은 도 (C6)")
print("\n🛑 종료: ESC 키")
print("=" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 웹캠을 열 수 없습니다.")
        break
    
    # 좌우 반전 (거울 모드)
    frame = cv2.flip(frame, 1)
    
    # BGR을 RGB로 변환 (Mediapipe용)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 손 인식
    result = hands.process(rgb)
    
    # 화면에 제목
    cv2.putText(frame, "Violin Orchestra - ESC to quit", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 손이 감지되면
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # 손 그리기
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 어떤 손가락이 펴져있는지 확인
            fingers = fingers_up(hand_landmarks)
            
            # 손 모양 판별
            current_gesture = None
            gesture_name = ""
            note_name = ""
            
            if fingers == [False, False, False, False, False]:
                current_gesture = 'fist'
                gesture_name = "Fist"
                note_name = "Do (C4)"
                
            elif fingers == [False, True, False, False, False]:
                current_gesture = 'one'
                gesture_name = "One"
                note_name = "Re (D4)"
                
            elif fingers == [False, True, True, False, False]:
                current_gesture = 'two'
                gesture_name = "Two"
                note_name = "Mi (E4)"
                
            elif fingers == [False, True, True, True, False]:
                current_gesture = 'three'
                gesture_name = "Three"
                note_name = "Fa (F4)"
                
            elif fingers == [False, True, True, True, True]:
                current_gesture = 'four'
                gesture_name = "Four"
                note_name = "Sol (G4)"
                
            elif fingers == [True, False, False, False, False]:
                current_gesture = 'thumb'
                gesture_name = "Thumb"
                note_name = "La (A4)"
                
            elif fingers == [True, True, False, False, False]:
                current_gesture = 'thumb_one'
                gesture_name = "Thumb+One"
                note_name = "Si (B4)"
                
            elif fingers == [True, True, True, True, True]:
                current_gesture = 'all'
                gesture_name = "All"
                note_name = "High Do (C5)"
            
            # 손 모양이 바뀌었을 때만 재생
            if current_gesture and current_gesture != previous_gesture:
                play_sound(current_gesture)
                previous_gesture = current_gesture
            
            # 화면에 현재 음계 표시
            if note_name:
                cv2.putText(frame, note_name, 
                           (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 100, 0), 4)
                cv2.putText(frame, f"({gesture_name})", 
                           (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 2)
    else:
        # 손이 없을 때
        cv2.putText(frame, "Show your hand!", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    
    # 화면 표시
    cv2.imshow("Violin Orchestra", frame)
    
    # ESC 키로 종료
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# 정리
cap.release()
cv2.destroyAllWindows()
pygame.quit()
print("\n👋 연주 종료! 수고하셨습니다!")
