import cv2
import mediapipe as mp
import pygame
import numpy as np

# pygame 초기화
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

# 음 재생 함수
def play_sound(frequency, duration=150):
    sample_rate = 44100
    n_samples = int(sample_rate * duration / 1000)
    t = np.linspace(0, duration/1000, n_samples, False)
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)
    wave_stereo = np.column_stack([wave, wave]).astype(np.float32)  # 스테레오로 변환
    sound = pygame.sndarray.make_sound((wave_stereo * 32767).astype(np.int16))
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
        fingers.append(tip_y < pip_y)  # 펴져있으면 True
    return fingers


while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

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
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
