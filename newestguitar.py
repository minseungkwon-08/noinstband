import cv2
import mediapipe as mp
import numpy as np
import pygame
import time

# ================== AUDIO SETTINGS ==================
SAMPLE_RATE = 44100
pygame.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=1024)
pygame.init()
channel = pygame.mixer.Channel(0)

AMPLITUDE = 0.2
FADE_MS = 8
NOTE_DURATION = 500  # 코드가 조금 더 길게 울림

# 음계 주파수
NOTES = {
    "C3": 130.81, "C#3": 138.59,
    "D3": 146.83, "D#3": 155.56,
    "E3": 164.81,
    "F3": 174.61, "F#3": 185.00,
    "G3": 196.00, "G#3": 207.65,
    "A3": 220.00, "A#3": 233.08,
    "B3": 246.94,
    "C4": 261.63, "C#4": 277.18,
    "D4": 293.66, "D#4": 311.13,
    "E4": 329.63,
    "F4": 349.23, "F#4": 369.99,
    "G4": 392.00, "G#4": 415.30,
    "A4": 440.00, "A#4": 466.16,
    "B4": 493.88,
    "C5": 523.25
}

# ================== SOUND FUNCTIONS ==================
def to_sound_array(mono_wave):
    stereo = np.column_stack([mono_wave, mono_wave])
    buf = np.ascontiguousarray((stereo * 32767.0).astype(np.int16))
    return buf

def apply_fade(wave, fade_ms=FADE_MS):
    n = len(wave)
    fade_len = max(1, int(SAMPLE_RATE * (fade_ms / 1000.0)))
    fade_in = np.linspace(0.0, 1.0, fade_len)
    fade_out = np.linspace(1.0, 0.0, fade_len)
    wave[:fade_len] *= fade_in
    wave[-fade_len:] *= fade_out
    return wave

# ================== SMOOTH GUITAR SYNTH ==================
def synth_guitar_smooth(freq, duration_ms=NOTE_DURATION, amplitude=AMPLITUDE):
    n_samples = int(SAMPLE_RATE * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n_samples, endpoint=False)
    
    # 기본 톤 + 오버톤
    wave = np.sin(2*np.pi*freq*t)           # 기본 주파수
    wave += 0.6 * np.sin(2*np.pi*freq*2*t)  # 2배음
    wave += 0.3 * np.sin(2*np.pi*freq*3*t)  # 3배음
    
    # 살짝 랜덤 노이즈 추가
    wave += 0.02 * np.random.randn(n_samples)
    
    # 부드러운 감쇠 + 진폭 흔들기
    env = np.exp(-3*t) * (1 + 0.05*np.sin(15*t))
    wave *= env * amplitude
    
    wave = apply_fade(wave)
    return to_sound_array(wave)

# ================== 기타 코드 사운드 ==================
CHORDS = {
    "C": ["C4", "E4", "G4"],
    "G": ["G3", "B3", "D4", "G4"],
    "Am": ["A3", "C4", "E4"],
    "F": ["F3", "A3", "C4"],
    "D": ["D4", "F#4", "A4"],
    "Em": ["E3", "G3", "B3"],
    "Dm": ["D4", "F4", "A4"]
}

def play_chord(chord_name):
    if chord_name not in CHORDS:
        return
    freqs = [NOTES[note] if note in NOTES else 0 for note in CHORDS[chord_name]]
    waves = [np.frombuffer(synth_guitar_smooth(f, NOTE_DURATION, AMPLITUDE), dtype=np.int16) for f in freqs if f > 0]

    max_len = max(len(w) for w in waves)
    mixed = np.zeros(max_len, dtype=np.float32)

    for w in waves:
        w_float = w.astype(np.float32)
        if len(w_float) < max_len:
            w_float = np.pad(w_float, (0, max_len - len(w_float)))
        mixed += w_float / len(waves)

    mixed = np.clip(mixed / np.max(np.abs(mixed)), -1, 1)
    sound = pygame.sndarray.make_sound(to_sound_array(mixed))
    channel.stop()
    channel.play(sound)

# ================== HAND TRACKING ==================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6)

def fingers_up(hand_landmarks):
    lm = hand_landmarks.landmark
    fingers = []

    thumb_up = lm[mp_hands.HandLandmark.THUMB_TIP].x < lm[mp_hands.HandLandmark.THUMB_IP].x
    fingers.append(thumb_up)
    
    for tip_id in (mp_hands.HandLandmark.INDEX_FINGER_TIP,
                   mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                   mp_hands.HandLandmark.RING_FINGER_TIP,
                   mp_hands.HandLandmark.PINKY_TIP):
        tip = lm[tip_id]
        pip = lm[tip_id - 2]
        fingers.append(tip.y < pip.y)
    return fingers

# ================== 손 모양 → 코드 매핑 ==================
def get_chord_from_fingers(fs):
    pattern_map = {
        (True, False, False, False, False): "C",
        (False, True, False, False, False): "G",
        (False, True, True, False, False): "Am",
        (False, True, True, True, False): "F",
        (False, True, True, True, True): "D",
        (False, False, False, False, False): "Em",
        (True, True, False, False, False): "Dm"
    }
    return pattern_map.get(tuple(fs), None)

# ================== MAIN LOOP ==================
cap = cv2.VideoCapture(0)
prev_chord = None
cooldown = 0.5
last_play = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        chord_name = None
        if res.multi_hand_landmarks:
            for hand_lms in res.multi_hand_landmarks:
                fs = fingers_up(hand_lms)
                chord_name = get_chord_from_fingers(fs)
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                cv2.putText(frame, f"Chord: {chord_name if chord_name else '?'}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        now = time.time()
        if chord_name and (chord_name != prev_chord or now - last_play > cooldown):
            play_chord(chord_name)
            prev_chord = chord_name
            last_play = now

        cv2.imshow("Gesture Guitar", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
