# gesture_orchestra.py
import cv2
import mediapipe as mp
import numpy as np
import pygame
import time

# ---------------- audio / synth settings ----------------
SAMPLE_RATE = 44100
pygame.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=1024)
pygame.init()
channel = pygame.mixer.Channel(0)

AMPLITUDE = 0.18    # 기본 진폭(작게하면 지지직 감소)
FADE_MS = 6
NOTE_DURATION = 300  # 기본 음 길이 (ms)

# note map (Hz)
NOTES = {
    "C4": 261.63, "D4": 293.66, "E4": 329.63, "F4": 349.23,
    "G4": 392.00, "A4": 440.00, "B4": 493.88, "C5": 523.25
}

# ---------------- utility: make stereo contiguous int16 ----------------
def to_sound_array(mono_wave):
    """mono_wave: np.array float (-1..1). returns stereo int16 contiguous"""
    stereo = np.column_stack([mono_wave, mono_wave])
    buf = np.ascontiguousarray((stereo * 32767.0).astype(np.int16))
    return buf

def apply_fade(wave, fade_ms=FADE_MS):
    n = len(wave)
    fade_len = max(1, int(SAMPLE_RATE * (fade_ms/1000.0)))
    fade_in = np.linspace(0.0, 1.0, fade_len)
    fade_out = np.linspace(1.0, 0.0, fade_len)
    wave[:fade_len] *= fade_in
    wave[-fade_len:] *= fade_out
    return wave

# ---------------- Piano-ish synth (simple additive + fast decay) ----------------
def synth_piano(freq, duration_ms=NOTE_DURATION, amplitude=AMPLITUDE):
    n = int(SAMPLE_RATE * duration_ms / 1000)
    t = np.linspace(0, duration_ms/1000, n, endpoint=False)
    # additive partials (few harmonics) with exponential decay
    wave = np.zeros_like(t)
    # weights & detune for more "percussive piano-like" tone
    partials = [(1.0,1.0), (0.6,2.0), (0.35,3.0), (0.2,4.0), (0.12,5.0)]
    env = np.exp(-6 * t)  # fast decay
    for w, mul in partials:
        wave += w * np.sin(2*np.pi*freq*mul*t)
    wave *= env * amplitude
    wave = apply_fade(wave, fade_ms=6)
    return to_sound_array(wave)

# ---------------- Karplus-Strong pluck (guitar-like) ----------------
def synth_guitar(freq, duration_ms=NOTE_DURATION, amplitude=AMPLITUDE):
    N = int(SAMPLE_RATE / freq)
    # generate noise burst
    n_samples = int(SAMPLE_RATE * duration_ms / 1000)
    # initial buffer (noise)
    buf = 0.5 * (2*np.random.rand(N) - 1.0)
    out = np.zeros(n_samples)
    # simple KS loop with averaging lowpass
    for i in range(n_samples):
        out[i] = buf[0]
        avg = 0.996 * 0.5 * (buf[0] + buf[1 % len(buf)])
        buf = np.append(buf[1:], [avg])
    # apply envelope (plucked)
    env = np.linspace(1.0, 0.0, n_samples)
    wave = amplitude * out * env
    wave = apply_fade(wave, fade_ms=8)
    return to_sound_array(wave)

# ---------------- Violin-like continuous tone (sustained + noise + vibrato) ----------------
def synth_violin(freq, duration_ms=NOTE_DURATION, amplitude=AMPLITUDE*0.9):
    n = int(SAMPLE_RATE * duration_ms / 1000)
    t = np.linspace(0, duration_ms/1000, n, endpoint=False)
    # base harmonic series (odd+even) with slow attack and vibrato
    vibrato = 1.0 + 0.0025 * np.sin(2*np.pi*5.5*t)  # small vibrato
    wave = np.zeros_like(t)
    # stronger lower partials, gentle higher ones
    for k, w in enumerate([1,2,3,4,5,6], start=1):
        wave += (0.6/(k)) * np.sin(2*np.pi*freq*k*t * vibrato)
    # slow attack to mimic bowing
    attack_len = int(0.02 * SAMPLE_RATE)
    env = np.ones(n)
    env[:attack_len] = np.linspace(0.0, 1.0, attack_len)
    # add subtle bow noise
    noise = 0.02 * (np.random.randn(n))
    wave = (wave * env + noise) * amplitude * 0.7
    wave = apply_fade(wave, fade_ms=6)
    return to_sound_array(wave)

# ---------------- play wrapper ----------------
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

# ---------------- Mediapipe hand setup ----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# ---------------- finger status helper ----------------
def fingers_up(hand_landmarks):
    thumbs = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x
    fingers = [thumbs]
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
last_time = 0
COOLDOWN = 0.12

print("Ready — press 1:piano  2:guitar  3:violin. ESC to quit.")

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

                # map finger combos to notes (same mapping as before)
                trigger = None
                # all closed = C4
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
                        play_note(NOTES[trigger])
                        prev_trigger = trigger
                        last_time = now

        cv2.imshow("Gesture Instruments", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord("1"):
            current_instrument = "piano"; print("Instrument = piano")
        elif key == ord("2"):
            current_instrument = "guitar"; print("Instrument = guitar")
        elif key == ord("3"):
            current_instrument = "violin"; print("Instrument = violin")

finally:
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
