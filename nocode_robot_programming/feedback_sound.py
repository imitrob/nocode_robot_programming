#!/usr/bin/env python3
import time
import threading
import queue
import numpy as np
import simpleaudio as sa # pip install simpleaudio

# Sound synthesis utilities
SAMPLE_RATE = 48000
MASTER_VOL = 0.6  # overall volume ceiling (0..1)

def _adsr_envelope(n, attack=0.01, release=0.15):
    """Simple AR envelope (in seconds)."""
    env = np.ones(n, dtype=np.float32)
    a = int(attack * SAMPLE_RATE)
    r = int(release * SAMPLE_RATE)
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
    if r > 0:
        env[-r:] = np.linspace(1.0, 0.0, r, dtype=np.float32)
    return env

def synth_tone(freq, duration, volume=0.3, harmonics=(1.0, 0.25, 0.12)):
    """
    Synthesize a gentle bell-like tone (sum of few sine harmonics) with an AR envelope.
    """
    n = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, n, endpoint=False, dtype=np.float32)

    # Sum a few harmonic partials (very soft) to avoid a harsh pure sine
    wave = np.zeros_like(t)
    for i, amp in enumerate(harmonics, start=1):
        wave += amp * np.sin(2 * np.pi * freq * i * t)

    # Normalize and envelope
    wave /= max(1.0, np.max(np.abs(wave)))
    wave *= _adsr_envelope(n, attack=0.008, release=min(0.18, duration * 0.6))
    wave *= (volume * MASTER_VOL)

    # Convert to int16 PCM
    return (wave * 32767).astype(np.int16)

def concat(*chunks, gaps=0.04):
    """Concatenate PCM chunks with short silences (gaps in seconds)."""
    if not chunks:
        return np.zeros(0, dtype=np.int16)
    silence = np.zeros(int(gaps * SAMPLE_RATE), dtype=np.int16)
    out = [chunks[0]]
    for c in chunks[1:]:
        out.extend([silence, c])
    return np.concatenate(out)

# Earcons: Musical notes (approx): E5=659.26 Hz, B5=987.77 Hz
E5 = 659.26
B5 = 987.77

START_CHIME = concat(
    synth_tone(E5, 0.18, volume=0.28),
    synth_tone(B5, 0.20, volume=0.28),
    gaps=0.05
)

END_CHIME = concat(
    synth_tone(B5, 0.18, volume=0.26),
    synth_tone(E5, 0.20, volume=0.26),
    gaps=0.05
)

# Optional super-soft heartbeat pip for "active" state (off by default)
HEARTBEAT_PIP = synth_tone(740.0, 0.12, volume=0.12)

# Audio worker (non-blocking)
_audio_q = queue.Queue()

def _audio_worker():
    while True:
        pcm = _audio_q.get()
        if pcm is None:
            break  # optional clean shutdown
        try:
            sa.play_buffer(pcm, 1, 2, SAMPLE_RATE).wait_done()
        except Exception:
            # Avoid crashing audio thread on device hiccups
            pass
        finally:
            _audio_q.task_done()

_audio_thread = threading.Thread(target=_audio_worker, daemon=True)
_audio_thread.start()

def play_pcm(pcm):
    """Queue a PCM buffer to the audio thread."""
    _audio_q.put(pcm)

# Integration with your 5 Hz loop
HEARTBEAT_ENABLED = False     # set True if you want the soft pip while active
HEARTBEAT_PERIOD_S = 2.0

def sound_thread(self):
    """ I use this function as a separate thread """
    check_period = 0.2  # 5 Hz
    last_state = None
    next_heartbeat = 0.0

    # print("Running 5 Hz control-state monitor. Press Ctrl+C to exit.")
    try:
        while True:
            controlled = self.teleop_has_control()

            # Rising edge: start sound
            if last_state is False and controlled is True:
                play_pcm(START_CHIME)
                next_heartbeat = time.time() + HEARTBEAT_PERIOD_S

            # Falling edge: end sound
            if last_state is True and controlled is False:
                play_pcm(END_CHIME)

            # Optional heartbeat while active (very soft and infrequent)
            if HEARTBEAT_ENABLED and controlled and time.time() >= next_heartbeat:
                play_pcm(HEARTBEAT_PIP)
                next_heartbeat = time.time() + HEARTBEAT_PERIOD_S

            last_state = controlled if last_state is not None else controlled
            # Initialize edge detection on first iteration
            if last_state is None:
                last_state = controlled

            time.sleep(check_period)
    except KeyboardInterrupt:
        pass
    finally:
        # Optional: stop audio thread cleanly
        _audio_q.put(None)
        _audio_thread.join(timeout=0.5)

if __name__ == "__main__":
    sound_thread()
