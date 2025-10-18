import cv2
import mediapipe as mp
import numpy as np
import pygame
import time

# ================== AUDIO / SYNTH SETTINGS ==================
SAMPLE_RATE = 44100
pygame.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=1024)
pygame.init()
channel = pygame.mixer.Channel(0)

AMPLITUDE = 0.18
FADE_MS = 6
NOTE_DURATION = 300

NOTES = {
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

def synth_piano(freq, duration_ms=NOTE_DURATION, amplitude=AMPLITUDE):
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

def synth_guitar(freq, duration_ms=NOTE_DURATION, amplitude=AMPLITUDE):
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

def synth_violin(freq, duration_ms=NOTE_DURATION, amplitude=AMPLITUDE * 0.9):
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

INSTRUMENTS = {
    "piano": synth_piano,
    "guitar": synth_guitar,
    "violin": synth_violin
}
current_instrument = "piano"

def play_note(freq):
    synth = INSTRUMENTS.get(current_instrument, synth_piano)
    buf = synth(freq)
    sound = pygame.sndarray.make_sound(buf)
    channel.stop()
    channel.play(sound)

# ================== MEDIAPIPE HAND SETUP ==================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.6)

# ================== FINGER STATUS DETECTION ==================
def fingers_up(hand_landmarks, handedness_label):
    """Detect which fingers are up. Works for both palm or back facing the camera."""
    lm = hand_landmarks.landmark

    # 손바닥/손등 구분하지 않고 엄지 방향 처리
    thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP].x
    thumb_ip = lm[mp_hands.HandLandmark.THUMB_IP].x
    if handedness_label == "Left":
        thumb_up = thumb_tip > thumb_ip  # 왼손은 오른쪽으로 벌리면 True
    else:
        thumb_up = thumb_tip < thumb_ip  # 오른손은 왼쪽으로 벌리면 True

    fingers = [thumb_up]
    for tip_id in (mp_hands.HandLandmark.INDEX_FINGER_TIP,
                   mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                   mp_hands.HandLandmark.RING_FINGER_TIP,
                   mp_hands.HandLandmark.PINKY_TIP):
        tip = lm[tip_id]
        pip = lm[tip_id - 2]
        fingers.append(tip.y < pip.y)
    return fingers

# ================== NOTE MAPPING ==================
def get_note_from_fingers(fs):
    note_map = {
        (False, False, False, False, False): "C4",
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

def get_freq_multiplier(fs):
    if fs == [False, False, False, False, False]:
        return 1.0
    elif fs == [False, True, False, False, False]:
        return 0.5
    elif fs == [True, True, True, True, True]:
        return 2.0
    else:
        return 1.0

# ================== MAIN LOOP ==================
cap = cv2.VideoCapture(0)
cooldowns = {"Left": 0, "Right": 0}
prev_triggers = {"Left": None, "Right": None}
COOLDOWN = 0.15

print("\n[Gesture Orchestra Ready 🎶]")
print("👉 Use both hands freely — palm/back doesn't matter.")
print("🎹 Right hand = pitch, Left hand = octave multiplier (x0.5 / x2)")
print("🎸 [1] Piano | [2] Guitar | [3] Violin\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        left_mult = 1.0
        right_note = None
        now = time.time()

        if res.multi_hand_landmarks and res.multi_handedness:
            for hand_lms, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                label = handed.classification[0].label
                fs = fingers_up(hand_lms, label)
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

                if label == "Left":
                    left_mult = get_freq_multiplier(fs)
                    txt = f"L:{''.join(['1' if f else '0' for f in fs])} x{left_mult}"
                    cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                else:
                    note = get_note_from_fingers(fs)
                    if note:
                        right_note = note
                    txt = f"R:{''.join(['1' if f else '0' for f in fs])} {note if note else '?'}"
                    cv2.putText(frame, txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # 사운드 재생
        if right_note:
            trigger = (right_note, left_mult)
            if trigger != prev_triggers["Right"] or (now - cooldowns["Right"]) > COOLDOWN:
                freq = NOTES[right_note] * left_mult
                play_note(freq)
                prev_triggers["Right"] = trigger
                cooldowns["Right"] = now

        cv2.imshow("Gesture Orchestra", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord("1"):
            current_instrument = "piano"; print("→ piano")
        elif key == ord("2"):
            current_instrument = "guitar"; print("→ guitar")
        elif key == ord("3"):
            current_instrument = "violin"; print("→ violin")

finally:
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
