# gesture_orchestra.py
# Gesture Orchestra — Guitar (Karplus-Strong) version
# 필요: python, opencv-python, mediapipe, numpy, pygame 설치

import cv2
import mediapipe as mp
import numpy as np
import pygame
import time

# ---------------- audio / synth settings ----------------
SAMPLE_RATE = 44100
pygame.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=1024)
pygame.init()
channel = pygame.mixer.Channel(0)  # 하나의 채널 재사용

AMPLITUDE = 0.16    # 기본 진폭 (0..1) — 필요하면 낮춰서 지직거림 감소
FADE_MS = 6
NOTE_DURATION_MS = 350  # 기본 음 길이 (ms)
COOLDOWN = 0.12         # 같은 제스처 반복 재생 방지(초)

# ---------------- note table ----------------
NOTES = {
    "C4": 261.63, "D4": 293.66, "E4": 329.63, "F4": 349.23,
    "G4": 392.00, "A4": 440.00, "B4": 493.88, "C5": 523.25
}

# ---------------- helper utilities ----------------
def to_sound_array(mono_wave):
    """mono_wave: float array (-1..1). returns stereo int16 contiguous"""
    stereo = np.column_stack([mono_wave, mono_wave])
    buf = np.ascontiguousarray((stereo * 32767.0).astype(np.int16))
    return buf

def apply_fade(wave, fade_ms=FADE_MS):
    n = len(wave)
    fade_len = max(1, int(SAMPLE_RATE * (fade_ms/1000.0)))
    if fade_len * 2 >= n:
        # 너무 짧은 음이면 전체에 완만한 창 적용
        fade_len = max(1, n // 10)
    fade_in = np.linspace(0.0, 1.0, fade_len)
    fade_out = np.linspace(1.0, 0.0, fade_len)
    wave[:fade_len] *= fade_in
    wave[-fade_len:] *= fade_out
    return wave

# ---------------- Improved Karplus-Strong guitar synth ----------------
def synth_guitar(freq, duration_ms=NOTE_DURATION_MS, amplitude=AMPLITUDE, damping=0.995, pick_noise=0.7):
    """
    Karplus-Strong 플럭(기타 비슷한 소리)을 생성해 stereo int16 배열 반환.
    freq: 주파수(Hz)
    duration_ms: 길이 (ms)
    amplitude: 볼륨
    damping: 루프 감쇠 (0.98..0.999)
    pick_noise: 초기 노이즈 크기 (0..1)
    """
    n_samples = int(SAMPLE_RATE * duration_ms / 1000)
    N = max(2, int(SAMPLE_RATE / freq))  # 루프 길이 (정수 샘플)

    # 1) 초기 노이즈 버퍼(플럭)
    init = (2.0 * np.random.rand(N) - 1.0) * pick_noise
    # 살짝 필터링해서 거친 잡음 완화
    init = np.convolve(init, [0.6, 0.4], mode='same')

    out = np.zeros(n_samples, dtype=np.float32)
    buf = init.copy()

    # 2) Karplus-Strong 루프
    idx = 0
    for i in range(n_samples):
        sample = buf[idx]
        out[i] = sample
        # 평균 + 감쇠 (간단한 저역 강조)
        next_val = damping * 0.5 * (buf[idx] + buf[(idx + 1) % N])
        buf[idx] = next_val
        idx = (idx + 1) % N

    # 3) 엔벨로프: 짧은 attack + 지수적 decay
    attack_len = max(1, int(0.005 * SAMPLE_RATE))  # 5ms attack
    env = np.ones(n_samples, dtype=np.float32)
    env[:attack_len] = np.linspace(0.0, 1.0, attack_len)
    # 지수 감쇠 (속도 조절)
    env *= np.exp(-3.0 * np.arange(n_samples) / SAMPLE_RATE)
    wave = out * env * amplitude

    # 4) 간단한 1-pole lowpass로 부드럽게
    alpha = 0.6
    lp = np.empty_like(wave)
    lp_prev = 0.0
    for i, v in enumerate(wave):
        lp_prev = alpha * lp_prev + (1.0 - alpha) * v
        lp[i] = lp_prev

    # 5) fade in/out 안전 처리
    lp = apply_fade(lp, fade_ms=6)

    # 6) stereo int16, contiguous
    buf_out = to_sound_array(lp)
    return buf_out

# ---------------- play wrapper for guitar ----------------
def play_note_guitar(freq):
    buf = synth_guitar(freq, duration_ms=NOTE_DURATION_MS, amplitude=AMPLITUDE, damping=0.995, pick_noise=0.7)
    sound = pygame.sndarray.make_sound(buf)
    channel.stop()
    channel.play(sound)

# ---------------- Mediapipe hand setup ----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# ---------------- finger status helper ----------------
def fingers_up(hand_landmarks):
    """엄지, 검지, 중지, 약지, 새끼 순서로 True(펴짐)/False(접힘) 반환"""
    # 엄지: x 비교 (화면은 좌우 반전되어 있음)
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip  = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_up = thumb_tip.x > thumb_ip.x
    fingers = [thumb_up]
    # 검지~새끼: tip.y < pip.y 면 펴짐
    for tip_id in (mp_hands.HandLandmark.INDEX_FINGER_TIP,
                   mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                   mp_hands.HandLandmark.RING_FINGER_TIP,
                   mp_hands.HandLandmark.PINKY_TIP):
        tip = hand_landmarks.landmark[tip_id]
        pip = hand_landmarks.landmark[tip_id - 2]
        fingers.append(tip.y < pip.y)
    return fingers  # [thumb, index, middle, ring, pinky]

# ---------------- Camera ----------------
cap = cv2.VideoCapture(0)
prev_trigger = None
last_time = 0.0

print("Gesture Guitar ready. Show gestures to play notes. ESC to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            for handLms in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                fs = fingers_up(handLms)  # [thumb, idx, mid, ring, pinky]

                trigger = None
                # mapping same as previous scheme:
                if fs == [False, False, False, False, False]:
                    trigger = "C4"
                elif fs == [False, True, False, False, False]:
                    trigger = "D4"
                elif fs == [False, True, True, False, False]:
                    trigger = "E4"
                elif fs == [False, True, True, True, False]:
                    trigger = "F4"
                elif fs == [False, True, True, True, True]:
                    trigger = "G4"
                elif fs == [True, False, False, False, False]:
                    trigger = "A4"
                elif fs == [True, True, False, False, False]:
                    trigger = "B4"
                elif fs == [True, True, True, True, True]:
                    trigger = "C5"

                if trigger is not None:
                    now = time.time()
                    if trigger != prev_trigger or (now - last_time) > COOLDOWN:
                        # play with guitar synth
                        play_note_guitar(NOTES[trigger])
                        prev_trigger = trigger
                        last_time = now

        cv2.imshow("Gesture Guitar", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
