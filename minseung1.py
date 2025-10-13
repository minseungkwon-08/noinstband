import cv2
import mediapipe as mp
import pygame
import numpy as np

pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

# 악기별 소리 생성 함수
def create_instrument_sound(frequency, duration, instrument='piano'):
    sample_rate = 44100
    n_samples = int(sample_rate * duration / 1000)
    t = np.linspace(0, duration/1000, n_samples, False)
    
    if instrument == 'piano':
        # 피아노: 기본음 + 약한 배음들
        wave = (np.sin(2 * np.pi * frequency * t) * 1.0 +
                np.sin(2 * np.pi * frequency * 2 * t) * 0.3 +
                np.sin(2 * np.pi * frequency * 3 * t) * 0.15)
        # ADSR 엔벨로프 (빠른 attack, 긴 decay)
        envelope = np.exp(-3 * t)
        
    elif instrument == 'violin':
        # 바이올린: 강한 홀수 배음
        wave = (np.sin(2 * np.pi * frequency * t) * 1.0 +
                np.sin(2 * np.pi * frequency * 3 * t) * 0.5 +
                np.sin(2 * np.pi * frequency * 5 * t) * 0.3 +
                np.sin(2 * np.pi * frequency * 7 * t) * 0.1)
        # 부드러운 attack
        attack = np.minimum(t * 50, 1.0)
        decay = np.exp(-1 * t)
        envelope = attack * decay
        
    elif instrument == 'guitar':
        # 기타: 풍부한 배음 + 빠른 감쇠
        wave = (np.sin(2 * np.pi * frequency * t) * 1.0 +
                np.sin(2 * np.pi * frequency * 2 * t) * 0.6 +
                np.sin(2 * np.pi * frequency * 3 * t) * 0.4 +
                np.sin(2 * np.pi * frequency * 4 * t) * 0.3)
        # 빠른 attack, 강한 decay
        envelope = np.exp(-5 * t)
        
    else:  # 기본 사인파
        wave = np.sin(2 * np.pi * frequency * t)
        envelope = 1.0
    
    # 엔벨로프 적용 및 정규화
    wave = wave * envelope
    wave = wave / np.max(np.abs(wave)) * 0.5  # 볼륨 조절
    
    wave_stereo = np.column_stack([wave, wave]).astype(np.float32)
    sound = pygame.sndarray.make_sound((wave_stereo * 32767).astype(np.int16))
    return sound

# 현재 악기 선택
current_instrument = 'piano'
instruments = ['piano', 'violin', 'guitar', 'sine']
instrument_index = 0

def play_sound(frequency, duration=300):
    sound = create_instrument_sound(frequency, duration, current_instrument)
    sound.play()

# Mediapipe 손 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

def fingers_up(hand_landmarks):
    fingers = []
    # 엄지
    if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
        fingers.append(True)
    else:
        fingers.append(False)
    # 검지~새끼
    for id in [mp_hands.HandLandmark.INDEX_FINGER_TIP,
               mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
               mp_hands.HandLandmark.RING_FINGER_TIP,
               mp_hands.HandLandmark.PINKY_TIP]:
        tip_y = hand_landmarks.landmark[id].y
        pip_y = hand_landmarks.landmark[id - 2].y
        fingers.append(tip_y < pip_y)
    return fingers

print("악기 변경: 'p'=피아노, 'v'=바이올린, 'g'=기타, 's'=사인파")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    
    # 현재 악기 표시
    cv2.putText(frame, f"Instrument: {current_instrument.upper()}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            fingers = fingers_up(handLms)
            
            # 손 모양별 음 매핑
            if fingers == [False, False, False, False, False]:
                play_sound(261)  # 도
            elif fingers == [False, True, False, False, False]:
                play_sound(293)  # 레
            elif fingers == [False, True, True, False, False]:
                play_sound(329)  # 미
            elif fingers == [False, True, True, True, False]:
                play_sound(349)  # 파
            elif fingers == [False, True, True, True, True]:
                play_sound(392)  # 솔
            elif fingers == [True, False, False, False, False]:
                play_sound(440)  # 라
            elif fingers == [True, True, False, False, False]:
                play_sound(493)  # 시
            elif fingers == [True, True, True, True, True]:
                play_sound(523)  # 높은 도
    
    cv2.imshow("Gesture Orchestra", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('p'):
        current_instrument = 'piano'
        print("악기 변경: 피아노")
    elif key == ord('v'):
        current_instrument = 'violin'
        print("악기 변경: 바이올린")
    elif key == ord('g'):
        current_instrument = 'guitar'
        print("악기 변경: 기타")
    elif key == ord('s'):
        current_instrument = 'sine'
        print("악기 변경: 사인파")

cap.release()
cv2.destroyAllWindows()
pygame.quit()
