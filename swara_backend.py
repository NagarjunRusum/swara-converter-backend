#!/usr/bin/env python3
"""
Swara Converter Backend
Upload a song (ideally vocals) -> detect the melody -> auto-detect Sa -> swaras + PDF.
"""

import base64
import hashlib
import logging
import os
import subprocess
import tempfile
import threading
import time
import uuid

import librosa
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy.signal import medfilt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_UPLOAD_MB = 60
MAX_SECONDS = 480          # longer songs risk exhausting a small server
ALLOWED_EXT = {'.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac'}
SR = 16000
HOP = 640                  # 40 ms per analysis frame
CHUNK_SEC = 15             # pitch is tracked this many seconds at a time to limit memory
SILENCE_LEVEL = 0.03       # under 3% of the loudest moment counts as silence
SILENCE_SKIP_SEC = 1.0     # silent stretches at least this long are not analysed at all
# n_thresholds 50 is ~1.3x faster than the default 100 with no loss of notes. Coarser
# `resolution` (0.2) is ~3x faster but drops many notes, so it is deliberately not used.
PYIN_SETTINGS = dict(fmin=65, fmax=1050, sr=SR, frame_length=2048, hop_length=HOP, n_thresholds=50)
REGLUE_GAP_SEC = 0.05     # same-pitch notes closer than this are one note (a glitch, not a repeat)
MIN_NOTE_SEC = 0.08        # shorter blips are treated as noise
JOB_TTL_SEC = 1800
PITCH_VERSION = 'p3'       # bump when the pitch-tracking settings below change
CACHE_DIR = os.path.join(tempfile.gettempdir(), 'swara_pitch_cache')
CACHE_MAX_FILES = 20

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024
CORS(app)

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
# Semitones above Sa -> (swara, marked?). A black key keeps the name of the swara just to
# its RIGHT (its komal / tivra form) with a '#' added: re#, ga#, ma#, dha#, ni#.
SWARA_AND_SHARP = [('sa', False), ('re', True), ('re', False), ('ga', True), ('ga', False),
                   ('ma', False), ('ma', True), ('pa', False), ('dha', True), ('dha', False),
                   ('ni', True), ('ni', False)]
HIGHER_FORM = {'sa': 's\u00e0', 're': 'r\u00e8', 'ga': 'g\u00e0', 'ma': 'm\u00e0',
               'pa': 'p\u00e0', 'dha': 'dh\u00e0', 'ni': 'n\u00ec'}  # grave accent = higher octave
PHRASE_GAP_SEC = 0.8       # a pause this long starts a new phrase
MAX_HOLD_DASHES = 4

# Krumhansl-Kessler key profiles, used to guess the home note (Sa)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

JOBS = {}
JOBS_LOCK = threading.Lock()


class UserError(Exception):
    """An error whose message is safe and useful to show to the user."""


def decode_audio(path):
    """Decode any audio file to 16 kHz mono floats with ffmpeg (far lighter on memory than librosa.load)."""
    cmd = ['ffmpeg', '-v', 'error', '-i', path, '-t', str(MAX_SECONDS + 1),
           '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-']
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
    except (subprocess.TimeoutExpired, OSError):
        raise UserError("Couldn't read that file. Please upload an mp3, wav, m4a, ogg or flac.")
    if result.returncode != 0 or len(result.stdout) < SR * 4:
        raise UserError("Couldn't read that file. Please upload an mp3, wav, m4a, ogg or flac.")
    return np.frombuffer(result.stdout, dtype=np.float32)


def _cache_path(audio_path):
    digest = hashlib.sha256()
    with open(audio_path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return os.path.join(CACHE_DIR, f'{PITCH_VERSION}-{digest.hexdigest()[:32]}.npz')


def _load_cached_pitch(path):
    try:
        with np.load(path) as data:
            return data['f0'], data['voiced'], data['rms']
    except (OSError, ValueError, KeyError):
        return None


def _save_cached_pitch(path, f0, voiced, rms):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(path, f0=f0, voiced=voiced, rms=rms)
        files = sorted((os.path.join(CACHE_DIR, n) for n in os.listdir(CACHE_DIR)), key=os.path.getmtime)
        for old in files[:-CACHE_MAX_FILES]:
            os.remove(old)
    except OSError:
        logger.warning("Could not write pitch cache")


def track_pitch(audio_path, report):
    """The slow step: returns per-frame pitch, voiced flags and loudness (cached per file)."""
    cache_path = _cache_path(audio_path)
    cached = _load_cached_pitch(cache_path)
    if cached is not None:
        report(0.95)
        return cached

    y = decode_audio(audio_path)
    report(0.05)
    if len(y) > MAX_SECONDS * SR:
        raise UserError(f"That song is longer than {MAX_SECONDS // 60} minutes. Please upload a shorter one.")

    # Loudness per frame: a dip inside a held pitch means the same note was sung again
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0]

    # Skip long silent stretches (instrumental gaps, pauses): nothing to detect there
    n_frames = len(rms)
    silent = rms < SILENCE_LEVEL * rms.max()
    min_run = int(SILENCE_SKIP_SEC * SR / HOP)
    skip = np.zeros(n_frames, dtype=bool)
    run_start = None
    for i in range(n_frames + 1):
        if i < n_frames and silent[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and i - run_start >= min_run:
                skip[run_start:i] = True
            run_start = None

    # Track pitch in short pieces: doing a whole song at once needs well over 1 GB of memory
    piece_frames = CHUNK_SEC * SR // HOP
    pieces = []
    i = 0
    while i < n_frames:
        if skip[i]:
            i += 1
            continue
        j = i
        while j < n_frames and not skip[j] and j - i < piece_frames:
            j += 1
        pieces.append((i, j))
        i = j
    f0 = np.full(n_frames, np.nan)
    voiced = np.zeros(n_frames, dtype=bool)
    total_frames = max(1, sum(b - a for a, b in pieces))
    done = 0
    for a, b in pieces:
        piece = y[a * HOP:b * HOP]
        if len(piece) < 4096:
            piece = np.pad(piece, (0, 4096 - len(piece)))
        f0_p, voiced_p, _ = librosa.pyin(piece, **PYIN_SETTINGS)
        f0[a:b] = f0_p[:b - a]
        voiced[a:b] = voiced_p[:b - a]
        done += b - a
        report(0.05 + 0.90 * done / total_frames)
    del y
    _save_cached_pitch(cache_path, f0, voiced, rms)
    return f0, voiced, rms


def extract_notes(audio_path, on_progress=None):
    """Return a list of (start_sec, end_sec, midi_int) notes for the sung melody."""
    report = on_progress or (lambda fraction: None)
    f0, voiced, rms = track_pitch(audio_path, report)
    good = voiced & ~np.isnan(f0)
    if good.sum() < 20:
        raise UserError("Couldn't find a clear melody in this audio. A vocals-only file works best.")

    # Correct for songs that aren't tuned exactly to A=440
    tuning = librosa.pitch_tuning(f0[good])
    midi = np.full(len(f0), np.nan)
    midi[good] = librosa.hz_to_midi(f0[good]) - tuning

    # Median-smooth each continuous sung stretch to remove pitch jitter
    padded = np.concatenate([[False], good, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    for start, end in zip(edges[::2], edges[1::2]):
        if end - start >= 5:
            midi[start:end] = medfilt(midi[start:end], 5)

    frame_sec = HOP / SR
    rms = np.concatenate([rms, np.zeros(max(0, len(midi) - len(rms)))])[:len(midi)]

    def is_dip(i):
        """True if frame i is a clear loudness valley, i.e. the singer re-struck the same pitch."""
        left, right = rms[max(0, i - 5):i], rms[i + 1:i + 6]
        if len(left) < 2 or len(right) < 2:
            return False
        return rms[i] < 0.6 * min(left.max(), right.max())

    notes = []  # [start, end, midi, started_by_restrike]
    restrike = False
    for i, m in enumerate(midi):
        if np.isnan(m):
            continue
        q = int(round(m))
        t = i * frame_sec
        if notes and notes[-1][2] == q and t - notes[-1][1] <= frame_sec * 1.5:
            if is_dip(i) and t - notes[-1][0] >= 0.16:
                restrike = True      # the dip frame itself belongs to neither note
            elif restrike:
                notes.append([t, t + frame_sec, q, True])
                restrike = False
            else:
                notes[-1][1] = t + frame_sec
        else:
            restrike = False
            notes.append([t, t + frame_sec, q, False])

    notes = [n for n in notes if n[1] - n[0] >= MIN_NOTE_SEC]
    # Re-merge same-pitch neighbours separated only by a dropped blip
    merged = []
    for n in notes:
        if merged and merged[-1][2] == n[2] and not n[3] and n[0] - merged[-1][1] <= REGLUE_GAP_SEC:
            merged[-1][1] = n[1]
        else:
            merged.append(n)
    if len(merged) < 3:
        raise UserError("Couldn't find enough notes in this audio. A vocals-only file works best.")
    return fold_octaves(merged)


def fold_octaves(notes):
    """Keep the melody in one octave range.

    Men and women often sing the same tune an octave apart. Inside a phrase, a big
    jump is moved to whichever octave lands nearest the previous note; then each
    whole phrase is moved to the octave nearest the song's usual range.
    """
    phrases, current = [], [list(notes[0][:3])]
    for prev, cur in zip(notes, notes[1:]):
        note = list(cur[:3])
        if note[0] - prev[1] >= PHRASE_GAP_SEC:
            phrases.append(current)
            current = []
        current.append(note)
    phrases.append(current)

    for ph in phrases:
        for i in range(1, len(ph)):
            prev, cur = ph[i - 1], ph[i]
            jump = cur[2] - prev[2]
            if abs(jump) < 10:
                continue
            # A real octave leap stays up there; a voice mix-up is a spike that falls back.
            nxt = ph[i + 1] if i + 1 < len(ph) else None
            is_spike = nxt is not None and abs(nxt[2] - prev[2]) < abs(nxt[2] - cur[2])
            if is_spike:
                cur[2] -= 12 * round(jump / 12)

    medians = np.array([np.median([n[2] for n in ph]) for ph in phrases])
    # reference range = the pitch that the most phrases sit within 6 semitones of
    counts = [np.sum(np.abs(medians - m) < 6) for m in medians]
    ref = float(np.median(medians[np.abs(medians - medians[int(np.argmax(counts))]) < 6]))
    out = []
    for ph, pm in zip(phrases, medians):
        shift = 12 * round((ref - pm) / 12)
        out.extend((n[0], n[1], n[2] + shift) for n in ph)
    return out


def detect_sa(notes):
    """Guess Sa (the home note) from which pitches the melody dwells on."""
    hist = np.zeros(12)
    for start, end, midi in notes:
        hist[midi % 12] += end - start
    best_score, best_pc = -2, 0
    for pc in range(12):
        rotated = np.roll(hist, -pc)  # rotated[0] is the candidate home note
        for profile in (MAJOR_PROFILE, MINOR_PROFILE):
            score = np.corrcoef(rotated, profile)[0, 1]
            if score > best_score:
                best_score, best_pc = score, pc
    return best_pc, float(best_score)


def label_notes(notes, sa_pc):
    """Turn midi notes into Indian-notation swaras, with octave marks and hold dashes."""
    pitches = np.array([n[2] for n in notes])
    med = int(round(np.median(pitches)))
    # Middle octave = the Sa-to-Sa span that holds the most notes (ties: the one nearest the median)
    candidates = [c for c in range(med - 12, med + 13) if (c - sa_pc) % 12 == 0]
    sa_ref = max(candidates, key=lambda c: (int(np.sum((pitches >= c) & (pitches < c + 12))), -abs(c - med)))
    unit = max(0.1, float(np.median([n[1] - n[0] for n in notes])))  # a "normal" note length
    out = []
    for start, end, midi in notes:
        octave, offset = divmod(midi - sa_ref, 12)
        octave = max(-1, min(1, octave))
        base, sharp = SWARA_AND_SHARP[offset]
        text = HIGHER_FORM[base] if octave > 0 else base
        if sharp:
            text += '#'
        if octave < 0:
            text += '.'
        dashes = max(0, min(MAX_HOLD_DASHES, round((end - start) / unit) - 1))
        out.append({
            'text': text,
            'dashes': dashes,
            'swara': text + (' ' + ' '.join(['\u2014'] * dashes) if dashes else ''),
            'keyboard': NOTE_NAMES[offset],                    # key to press, with Sa = C
            'midi': 60 + offset + 12 * octave,                 # same, for MIDI playback (Sa = middle C)
            'real_keyboard': NOTE_NAMES[midi % 12],            # the pitch actually sung in the song
            'real_midi': int(midi),
            'offset': int(offset),
            'time': round(float(start), 2),
            'duration': round(float(end - start), 2),
        })
    return out


def keyboard_transpose(sa_pc):
    """Semitones to transpose the keyboard so that playing Sa as C sounds at the song's Sa."""
    return sa_pc if sa_pc <= 6 else sa_pc - 12


def play_instruction(sa_name, transpose):
    if transpose == 0:
        return "Play with sa = C on the keyboard (this song's Sa is also C, so no transpose needed)."
    direction = 'up' if transpose > 0 else 'down'
    return (f"Play with sa = C on the keyboard. This song's real Sa is {sa_name}: "
            f"set your keyboard's TRANSPOSE to {transpose:+d} ({abs(transpose)} semitone"
            f"{'s' if abs(transpose) > 1 else ''} {direction}) so it sounds like the song.")


def generate_pdf(items, sa_name, transpose):
    """Build the practice sheet PDF and return its bytes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'swaras.pdf')
        doc = SimpleDocTemplate(path, pagesize=A4, topMargin=0.5 * inch, bottomMargin=0.5 * inch)
        base = getSampleStyleSheet()
        story = [
            Paragraph("Swara Notation", ParagraphStyle('T', parent=base['Heading1'], fontSize=24,
                                                      alignment=TA_CENTER, spaceAfter=6)),
            Paragraph(play_instruction(sa_name, transpose), ParagraphStyle(
                'S', parent=base['Normal'], fontSize=12, leading=16, alignment=TA_CENTER,
                textColor=colors.HexColor('#444444'))),
            Spacer(1, 0.2 * inch),
        ]
        line_style = ParagraphStyle('L', parent=base['Normal'], fontSize=18, fontName='Courier-Bold',
                                    alignment=TA_CENTER, leading=22, spaceAfter=2)
        key_style = ParagraphStyle('K', parent=base['Normal'], fontSize=10, textColor=colors.grey,
                                   alignment=TA_CENTER, spaceAfter=16)
        def token(c):
            return c['text'] + ''.join('&nbsp;\u2014' for _ in range(c['dashes']))

        # Fill each row up to about 38 characters so a row never wraps, and the keyboard
        # letters underneath always line up with the notes above them
        rows, row, width = [], [], 0
        for c in items:
            w = len(c['text']) + 2 * c['dashes'] + 1
            if row and width + w > 38:
                rows.append(row)
                row, width = [], 0
            row.append(c)
            width += w
        if row:
            rows.append(row)
        for chunk in rows:
            story.append(Paragraph(' '.join(token(c) for c in chunk), line_style))
            story.append(Paragraph('(' + ' '.join(c['keyboard'] for c in chunk) + ')', key_style))
        story.append(Spacer(1, 0.3 * inch))
        story.append(Paragraph(
            "<b>How to read this:</b> sa. = lower octave | sa = middle octave | s\u00e0 = higher octave<br/>"
            "A # after a swara (re#, ga#, ma#, dha#, ni#) means the black key just before (left of) that swara's key.<br/>"
            "A dash (\u2014) after a swara means hold that note for longer.<br/>"
            "The grey letters under each line are the keys to press, counting sa as C.",
            ParagraphStyle('G', parent=base['Normal'], fontSize=10, textColor=colors.HexColor('#666666'),
                           leading=14)))
        doc.build(story)
        with open(path, 'rb') as f:
            return f.read()


def process_audio(path, sa_override, set_stage):
    set_stage('Listening for the melody...', 0.02)
    notes = extract_notes(path, lambda fraction: set_stage(None, fraction))
    set_stage('Working out Sa and the swaras...', 0.96)
    if sa_override:
        sa_pc, confidence = NOTE_NAMES.index(sa_override), None
    else:
        sa_pc, confidence = detect_sa(notes)
    items = label_notes(notes, sa_pc)
    sa_name = NOTE_NAMES[sa_pc]
    set_stage('Building your PDF...', 0.98)
    transpose = keyboard_transpose(sa_pc)
    pdf = generate_pdf(items, sa_name, transpose)
    return {
        'swaras': [i['swara'] for i in items],
        'notes': items,
        'count': len(items),
        'duration': items[-1]['time'],
        'sa': sa_name,
        'transpose': transpose,
        'sa_auto': sa_override is None,
        'sa_confidence': None if confidence is None else round(confidence, 2),
        'pdf_base64': base64.b64encode(pdf).decode('ascii'),
    }


def run_job(job_id, path, sa_override):
    def set_stage(text, progress=None):
        with JOBS_LOCK:
            if text:
                JOBS[job_id]['stage'] = text
            if progress is not None:
                JOBS[job_id]['progress'] = round(progress, 3)
    try:
        result = process_audio(path, sa_override, set_stage)
        with JOBS_LOCK:
            JOBS[job_id].update(status='done', result=result)
    except UserError as e:
        with JOBS_LOCK:
            JOBS[job_id].update(status='error', error=str(e))
    except Exception:
        logger.exception("Job failed")
        with JOBS_LOCK:
            JOBS[job_id].update(status='error', error="Something went wrong while processing this song. Please try another file.")
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def purge_old_jobs():
    cutoff = time.time() - JOB_TTL_SEC
    with JOBS_LOCK:
        for jid in [j for j, v in JOBS.items() if v['created'] < cutoff]:
            del JOBS[jid]


@app.route('/api/convert', methods=['POST'])
def convert():
    """Accept an audio upload and start a background job."""
    purge_old_jobs()
    upload = request.files.get('file')
    if not upload or not upload.filename:
        return jsonify({'error': 'Please choose an audio file to upload.'}), 400
    ext = os.path.splitext(upload.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({'error': 'Please upload an mp3, wav, m4a, ogg or flac file.'}), 400
    sa_override = (request.form.get('sa') or '').strip().upper() or None
    if sa_override and sa_override not in NOTE_NAMES:
        return jsonify({'error': 'Unknown Sa note.'}), 400

    fd, path = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    upload.save(path)
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {'status': 'running', 'stage': 'Starting...', 'progress': 0.0, 'created': time.time()}
    threading.Thread(target=run_job, args=(job_id, path, sa_override), daemon=True).start()
    return jsonify({'job_id': job_id}), 202


@app.route('/api/jobs/<job_id>', methods=['GET'])
def job_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({'error': 'Job not found. It may have expired - please upload again.'}), 404
        body = {'status': job['status'], 'stage': job.get('stage'), 'progress': job.get('progress', 0.0)}
        if job['status'] == 'done':
            body['result'] = job['result']
        elif job['status'] == 'error':
            body['error'] = job['error']
    return jsonify(body)


@app.errorhandler(413)
def too_large(_):
    return jsonify({'error': f'That file is too large. The limit is {MAX_UPLOAD_MB} MB.'}), 413


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=port)
