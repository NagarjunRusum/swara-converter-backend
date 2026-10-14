#!/usr/bin/env python3
"""
Swara Converter Backend
Upload a song (ideally vocals) -> detect the melody -> auto-detect Sa -> swaras + PDF.
"""

import base64
import logging
import os
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
MIN_NOTE_SEC = 0.08        # shorter blips are treated as noise
JOB_TTL_SEC = 1800

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024
CORS(app)

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
# Semitones above Sa -> swara (b = komal/flat, # = tivra/sharp)
SWARA_BY_OFFSET = ['Sa', 'Reb', 'Re', 'Gab', 'Ga', 'Ma', 'Ma#', 'Pa', 'Dhab', 'Dha', 'Nib', 'Ni']

# Krumhansl-Kessler key profiles, used to guess the home note (Sa)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

JOBS = {}
JOBS_LOCK = threading.Lock()


class UserError(Exception):
    """An error whose message is safe and useful to show to the user."""


def extract_notes(audio_path):
    """Return a list of (start_sec, end_sec, midi_int) notes for the sung melody."""
    try:
        duration = librosa.get_duration(path=audio_path)
    except Exception:
        raise UserError("Couldn't read that file. Please upload an mp3, wav, m4a, ogg or flac.")
    if duration > MAX_SECONDS:
        raise UserError(f"That song is {duration / 60:.1f} minutes long. The limit is {MAX_SECONDS // 60} minutes.")

    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=1050, sr=SR, frame_length=2048, hop_length=HOP)
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
    notes = []  # [start, end, midi]
    for i, m in enumerate(midi):
        if np.isnan(m):
            continue
        q = int(round(m))
        t = i * frame_sec
        if notes and notes[-1][2] == q and t - notes[-1][1] <= frame_sec * 1.5:
            notes[-1][1] = t + frame_sec
        else:
            notes.append([t, t + frame_sec, q])

    notes = [n for n in notes if n[1] - n[0] >= MIN_NOTE_SEC]
    # Re-merge same-pitch neighbours separated only by a dropped blip
    merged = []
    for n in notes:
        if merged and merged[-1][2] == n[2] and n[0] - merged[-1][1] <= 0.15:
            merged[-1][1] = n[1]
        else:
            merged.append(n)
    if len(merged) < 3:
        raise UserError("Couldn't find enough notes in this audio. A vocals-only file works best.")
    return merged


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
    """Turn midi notes into swara labels, with octave shown by letter case."""
    pitches = np.array([n[2] for n in notes])
    med = int(round(np.median(pitches)))
    sa_ref = med - ((med - sa_pc) % 12)  # Sa at or just below the typical sung pitch
    out = []
    for start, end, midi in notes:
        rel = midi - sa_ref
        octave, offset = divmod(rel, 12)
        name = SWARA_BY_OFFSET[offset]
        if octave < 0:
            name = name.lower()
        elif octave > 0:
            name = name.upper()
        out.append({
            'swara': name,
            'keyboard': NOTE_NAMES[midi % 12],
            'midi': int(midi),
            'offset': int(offset),
            'time': round(float(start), 2),
            'duration': round(float(end - start), 2),
        })
    return out


def generate_pdf(items, sa_name):
    """Build the practice sheet PDF and return its bytes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'swaras.pdf')
        doc = SimpleDocTemplate(path, pagesize=A4, topMargin=0.5 * inch, bottomMargin=0.5 * inch)
        base = getSampleStyleSheet()
        story = [
            Paragraph("Swara Notation", ParagraphStyle('T', parent=base['Heading1'], fontSize=24,
                                                      alignment=TA_CENTER, spaceAfter=6)),
            Paragraph(f"Sa = {sa_name} on the keyboard", ParagraphStyle('S', parent=base['Normal'], fontSize=12,
                                                                      alignment=TA_CENTER, textColor=colors.HexColor('#444444'))),
            Spacer(1, 0.2 * inch),
        ]
        line_style = ParagraphStyle('L', parent=base['Normal'], fontSize=18, fontName='Courier-Bold',
                                    alignment=TA_CENTER, leading=22, spaceAfter=2)
        key_style = ParagraphStyle('K', parent=base['Normal'], fontSize=10, textColor=colors.grey,
                                   alignment=TA_CENTER, spaceAfter=16)
        per_line = 12
        for i in range(0, len(items), per_line):
            chunk = items[i:i + per_line]
            story.append(Paragraph(' '.join(c['swara'] for c in chunk), line_style))
            story.append(Paragraph('(' + ' '.join(c['keyboard'] for c in chunk) + ')', key_style))
        story.append(Spacer(1, 0.3 * inch))
        story.append(Paragraph(
            "<b>Notation guide:</b> lowercase (sa) = lower octave | Normal (Sa) = middle octave | "
            "UPPERCASE (SA) = higher octave<br/>b after a name = komal (flat), # = tivra (sharp), "
            "e.g. Gab = komal Ga. Grey letters below each line are the keyboard keys.",
            ParagraphStyle('G', parent=base['Normal'], fontSize=10, textColor=colors.HexColor('#666666'))))
        doc.build(story)
        with open(path, 'rb') as f:
            return f.read()


def process_audio(path, sa_override, set_stage):
    set_stage('Listening for the melody...')
    notes = extract_notes(path)
    set_stage('Working out Sa and the swaras...')
    if sa_override:
        sa_pc, confidence = NOTE_NAMES.index(sa_override), None
    else:
        sa_pc, confidence = detect_sa(notes)
    items = label_notes(notes, sa_pc)
    sa_name = NOTE_NAMES[sa_pc]
    set_stage('Building your PDF...')
    pdf = generate_pdf(items, sa_name)
    return {
        'swaras': [i['swara'] for i in items],
        'notes': items,
        'count': len(items),
        'duration': items[-1]['time'],
        'sa': sa_name,
        'sa_auto': sa_override is None,
        'sa_confidence': None if confidence is None else round(confidence, 2),
        'pdf_base64': base64.b64encode(pdf).decode('ascii'),
    }


def run_job(job_id, path, sa_override):
    def set_stage(text):
        with JOBS_LOCK:
            JOBS[job_id]['stage'] = text
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
        JOBS[job_id] = {'status': 'running', 'stage': 'Starting...', 'created': time.time()}
    threading.Thread(target=run_job, args=(job_id, path, sa_override), daemon=True).start()
    return jsonify({'job_id': job_id}), 202


@app.route('/api/jobs/<job_id>', methods=['GET'])
def job_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({'error': 'Job not found. It may have expired - please upload again.'}), 404
        body = {'status': job['status'], 'stage': job.get('stage')}
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
