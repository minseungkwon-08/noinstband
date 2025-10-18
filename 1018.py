#필요한 라이브러리 가져옴. 컴퓨터 내에서 pip install ~~(버전명은 requirements.txt에 있음)해서 설치한 후 그 환경에서 실행해야 함!
import cv2
import mediapipe as mp
import numpy as np
import pygame
import time

# ================== AUDIO / SYNTH SETTINGS ==================
SAMPLE_RATE = 44100
pygame.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=1024) #pygame을 이용해 오디오 재생 / 이 부분은 잘 모름.. 
pygame.init()
channel = pygame.mixer.Channel(0)

AMPLITUDE = 0.18
FADE_MS = 6
NOTE_DURATION = 300

NOTES = { #음계 주파수 지정
    "C4": 261.63, "C#4": 277.18,
    "D4": 293.66, "D#4": 311.13,
    "E4": 329.63,
    "F4": 349.23, "F#4": 369.99,
    "G4": 392.00, "G#4": 415.30,
    "A4": 440.00, "A#4": 466.16,
    "B4": 493.88,
    "C5": 523.25
}

# ================== SOUND UTILITIES ==================
def to_sound_array(mono_wave): #음파 만드는 함수 - 1채널 음파를 2채널로 변환 -> pygame에서 재생할 수 있는 형식
    stereo = np.column_stack([mono_wave, mono_wave])
    buf = np.ascontiguousarray((stereo * 32767.0).astype(np.int16))
    return buf

def apply_fade(wave, fade_ms=FADE_MS): #음의 시작과 끝을 부드럽게 만드는 함수
    n = len(wave)
    fade_len = max(1, int(SAMPLE_RATE * (fade_ms / 1000.0)))
    fade_in = np.linspace(0.0, 1.0, fade_len)
    fade_out = np.linspace(1.0, 0.0, fade_len)
    wave[:fade_len] *= fade_in
    wave[-fade_len:] *= fade_out
    return wave

def synth_piano(freq, duration_ms=NOTE_DURATION, amplitude=AMPLITUDE): #지수 감소 함수를 이용해서 피아노와 비슷한 음 구현
    n = int(SAMPLE_RATE * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n, endpoint=False)
    wave = np.zeros_like(t)
    partials = [(1.0, 1.0), (0.6, 2.0), (0.35, 3.0), (0.2, 4.0), (0.12, 5.0)]
    env = np.exp(-6 * t)
    for w, mul in partials:
        wave += w * np.sin(2 * np.pi * freq * mul * t)
    wave *= env * amplitude
    wave = apply_fade(wave)
    return to_sound_array(wave)

def synth_guitar(freq, duration_ms=NOTE_DURATION, amplitude=AMPLITUDE): #karplus strong 알고리즘으로 기타 소리 흉내 : 잘 모르는 부분... 
    N = int(SAMPLE_RATE / freq)
    n_samples = int(SAMPLE_RATE * duration_ms / 1000)
    buf = 0.5 * (2 * np.random.rand(N) - 1.0)
    out = np.zeros(n_samples)
    for i in range(n_samples):
        out[i] = buf[0]
        avg = 0.996 * 0.5 * (buf[0] + buf[1 % len(buf)])
        buf = np.append(buf[1:], [avg])
    env = np.linspace(1.0, 0.0, n_samples)
    wave = amplitude * out * env
    wave = apply_fade(wave)
    return to_sound_array(wave)

def synth_violin(freq, duration_ms=NOTE_DURATION, amplitude=AMPLITUDE * 0.9): #역시 바이올린 소리 흉내
    n = int(SAMPLE_RATE * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n, endpoint=False)
    vibrato = 1.0 + 0.0025 * np.sin(2 * np.pi * 5.5 * t)
    wave = np.zeros_like(t)
    for k in range(1, 7):
        wave += (0.6 / k) * np.sin(2 * np.pi * freq * k * t * vibrato)
    attack_len = int(0.02 * SAMPLE_RATE)
    env = np.ones(n)
    env[:attack_len] = np.linspace(0.0, 1.0, attack_len)
    noise = 0.02 * np.random.randn(n)
    wave = (wave * env + noise) * amplitude * 0.7
    wave = apply_fade(wave)
    return to_sound_array(wave)

INSTRUMENTS = { #악기 이름을 함수에 연결
    "piano": synth_piano,
    "guitar": synth_guitar,
    "violin": synth_violin
}
current_instrument = "piano" #기본 시작 : 피아노

def play_note(freq): #연주하는 함수
    synth = INSTRUMENTS.get(current_instrument, synth_piano) #현재 악기에 해당하는 함수 선택 없으면 기본값으로 synth_pianot씀
    buf = synth(freq) #선택된 함수로 음파 생성
    sound = pygame.sndarray.make_sound(buf) #음파 배열을 pygameSound로 변환
    channel.stop()#이전 소리 멈추고
    channel.play(sound)#새 소리 재생

# mediapipe 손 인식 설정
mp_hands = mp.solutions.hands 
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.6) #손 인식 모델 생성 -> 최대 2개 손 인식 / 신뢰도 70, 60

# 손가락 접음/핌 감지
def fingers_up(hand_landmarks, handedness_label):
    lm = hand_landmarks.landmark #손의 21개 지점 좌표(mediapipe로 인식)

    # 손바닥/손등 구분하지 않고 엄지 방향 처리
    thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP].x #엄지손가락 끝 관절 x좌표 추출
    thumb_ip = lm[mp_hands.HandLandmark.THUMB_IP].x #엄지손가락 두번째 관절 x좌표 추출
    if handedness_label == "Left":
        thumb_up = thumb_tip > thumb_ip  # 왼손은 오른쪽으로 벌리면 True
    else:
        thumb_up = thumb_tip < thumb_ip  # 오른손은 왼쪽으로 벌리면 True

    fingers = [thumb_up]
    for tip_id in (mp_hands.HandLandmark.INDEX_FINGER_TIP, #나머지 3개는 y좌로 판단
                   mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                   mp_hands.HandLandmark.RING_FINGER_TIP,
                   mp_hands.HandLandmark.PINKY_TIP):
        tip = lm[tip_id]
        pip = lm[tip_id - 2]
        fingers.append(tip.y < pip.y) #끝이 관절보다 위에 있으면 펼친 것
    return fingers #[thumb, indexx, middle, pinky]배열 반환

# 음 짝 지어주기
def get_note_from_fingers(fs):
    note_map = {
        (False, False, False, False, False): "C4", #손가락 패턴을 음 이름으로 변환 ex) falsefalsefalsefalsetrue는 소지만 핀 것 -> 도샾
        (False, False, False, False, True): "C#4",
        (False, True, False, False, False): "D4",
        (False, True, False, False, True): "D#4",
        (False, True, True, False, False): "E4",
        (False, True, True, True, False): "F4",
        (False, True, True, True, True): "F#4",
        (True, False, False, False, False): "G4",
        (True, False, False, False, True): "G#4",
        (True, True, False, False, False): "A4",
        (True, True, False, False, True): "A#4",
        (True, True, True, False, False): "B4",
    }
    return note_map.get(tuple(fs), None)
#왼손으로 옥타브 변환, 주먹이면 그대로, 검지만 피면 한옥타브 아래로, 다 피면 한옥타브 위로
def get_freq_multiplier(fs):
    if fs == [False, False, False, False, False]:
        return 1.0
    elif fs == [False, True, False, False, False]:
        return 0.5
    elif fs == [True, True, True, True, True]:
        return 2.0
    else:
        return 1.0

#메인 동작
cap = cv2.VideoCapture(0) #cap은 카메라 객체
cooldowns = {"Left": 0, "Right": 0} #손마다 마지막 재생 시간 기록
prev_triggers = {"Left": None, "Right": None}
COOLDOWN = 0.15 # 0.15초 이내 같은 음 다시 재생x



try:
    while True:
        ret, frame = cap.read() #카메라에서 한 프레임 읽기
        if not ret: #카메라 오류 시 반복문 탈출
            break
        frame = cv2.flip(frame, 1) #좌우 반전
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #BGR -> RGB 색상 변환
        res = hands.process(rgb) #mediapipe가 손 인식

        left_mult = 1.0 #기본값, 매 루프마다 초기화됨. 
        right_note = None
        now = time.time()

        if res.multi_hand_landmarks and res.multi_handedness: #인식된 손들의 좌표 리스트와 왼/오 정보를 함께 반복
            for hand_lms, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                label = handed.classification[0].label #left or right
                fs = fingers_up(hand_lms, label) #랜드마크와 왼/오
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS) #그림

                if label == "Left":
                    left_mult = get_freq_multiplier(fs) #왼손이면 배수 받아오기(함수에서)
                    txt = f"L:{''.join(['1' if f else '0' for f in fs])} x{left_mult}"
                    cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                else:
                    note = get_note_from_fingers(fs) #아닌경ㅇ  : 오른손이면 음 받아오기
                    if note:
                        right_note = note
                    txt = f"R:{''.join(['1' if f else '0' for f in fs])} {note if note else '?'}"
                    cv2.putText(frame, txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # 사운드 재생
        if right_note: #오른손이면 음 매핑
            trigger = (right_note, left_mult) 
            if trigger != prev_triggers["Right"] or (now - cooldowns["Right"]) > COOLDOWN: #음이 바뀌었거나 cooldown시간이 지났으면 
                freq = NOTES[right_note] * left_mult #최종 주파수 계산해서
                play_note(freq) #소리 재생
                prev_triggers["Right"] = trigger
                cooldowns["Right"] = now

        cv2.imshow("Gesture Orchestra", frame) #프레임을 화면에 표시, 1ms동안 키 입력 대기, ESC누르면 종료
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord("1"): #1은 피아노
            current_instrument = "piano"; print("→ piano")
        elif key == ord("2"): #2는 기타
            current_instrument = "guitar"; print("→ guitar")
        elif key == ord("3"): #3은 바이올린
            current_instrument = "violin"; print("→ violin")

finally:
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
