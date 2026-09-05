#!/usr/bin/env python3
"""
Swara Converter Backend
Converts YouTube audio to Indian classical swaras with keyboard mapping
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import tempfile
import subprocess
import numpy as np
import librosa
import soundfile as sf
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Swara definitions - frequency mapping (in cents from C0 at 16.35 Hz)
SWARA_MAP = {
    'Sa': 0,      # C
    'Re': 200,    # D
    'Ga': 400,    # E
    'Ma': 500,    # F
    'Pa': 700,    # G
    'Dha': 900,   # A
    'Ni': 1100,   # B
}

SWARA_TO_NOTE = {
    'Sa': 'C',
    'Re': 'D',
    'Ga': 'E',
    'Ma': 'F',
    'Pa': 'G',
    'Dha': 'A',
    'Ni': 'B',
}

def cents_to_hz(cents, base_freq=16.35):
    """Convert cents to Hz"""
    return base_freq * (2 ** (cents / 1200))

def hz_to_cents(freq, base_freq=16.35):
    """Convert Hz to cents from C0"""
    if freq <= 0:
        return None
    return 1200 * np.log2(freq / base_freq)

def freq_to_swara(freq, cents):
    """Map frequency to nearest swara with octave notation"""
    if freq is None or cents is None:
        return None
    
    # Get cents from C0
    c_cents = cents % 1200  # Normalize to one octave
    
    # Find nearest swara
    min_diff = float('inf')
    nearest_swara = None
    
    for swara, swara_cents in SWARA_MAP.items():
        diff = abs(c_cents - swara_cents)
        if diff > 600:  # Check wraparound
            diff = 1200 - diff
        if diff < min_diff:
            min_diff = diff
            nearest_swara = swara
    
    # Determine octave
    octave_num = int(cents / 1200)
    
    # Notation: lowercase = octave below, normal = middle, CAPS = octave above
    if octave_num < 4:
        notation = nearest_swara.lower()
    elif octave_num == 4:
        notation = nearest_swara
    else:
        notation = nearest_swara.upper()
    
    keyboard_note = SWARA_TO_NOTE[nearest_swara]
    
    return {
        'swara': notation,
        'swara_name': nearest_swara,
        'keyboard': keyboard_note,
        'frequency': freq,
        'confidence': 1.0 - (min_diff / 100)  # Rough confidence
    }

def extract_youtube_audio(youtube_url, output_path):
    """Extract audio from YouTube using yt-dlp"""
    try:
        cmd = [
            'yt-dlp',
            '-f', 'bestaudio',
            '-x',
            '--audio-format', 'wav',
            '-o', output_path,
            youtube_url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise Exception(f"yt-dlp error: {result.stderr}")
        return output_path
    except Exception as e:
        logger.error(f"YouTube extraction failed: {e}")
        raise

def detect_melodic_line(audio_path, sr=22050):
    """Detect the melodic line (dominant frequency) from audio"""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=sr)
        
        # Compute STFT for pitch tracking
        S = np.abs(librosa.stft(y))
        
        # Use librosa's pYIN for pitch detection (more robust)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=50,  # Minimum frequency (below Sa)
            fmax=2000,  # Maximum frequency
            sr=sr
        )
        
        # Get time frames
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr)
        
        # Convert to swaras
        swaras = []
        for i, (freq, is_voiced, confidence) in enumerate(zip(f0, voiced_flag, voiced_probs)):
            if is_voiced and freq > 0:
                cents = hz_to_cents(freq)
                swara_info = freq_to_swara(freq, cents)
                if swara_info:
                    swara_info['time'] = float(times[i])
                    swara_info['voiced_confidence'] = float(confidence)
                    swaras.append(swara_info)
        
        return swaras, sr
    except Exception as e:
        logger.error(f"Pitch detection failed: {e}")
        raise

def generate_pdf(swaras, output_path, title="Swara Notation"):
    """Generate PDF with swara notation"""
    try:
        doc = SimpleDocTemplate(output_path, pagesize=A4, 
                               topMargin=0.5*inch, bottomMargin=0.5*inch)
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=getSampleStyleSheet()['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Swaras in lines (like sheet music)
        # Group swaras in lines of ~16 per line for readability
        swaras_per_line = 16
        
        line_style = ParagraphStyle(
            'SwaraLine',
            parent=getSampleStyleSheet()['Normal'],
            fontSize=20,
            fontName='Courier-Bold',
            alignment=TA_CENTER,
            spaceAfter=12,
            leading=24
        )
        
        for i in range(0, len(swaras), swaras_per_line):
            chunk = swaras[i:i + swaras_per_line]
            # Create swara line
            swara_text = ' '.join([s['swara'] for s in chunk])
            story.append(Paragraph(swara_text, line_style))
            
            # Create keyboard mapping line below
            keyboard_text = ' '.join([s['keyboard'] for s in chunk])
            keyboard_style = ParagraphStyle(
                'Keyboard',
                parent=getSampleStyleSheet()['Normal'],
                fontSize=10,
                textColor=colors.grey,
                alignment=TA_CENTER,
                spaceAfter=20
            )
            story.append(Paragraph(f"({keyboard_text})", keyboard_style))
        
        # Add legend
        story.append(Spacer(1, 0.3*inch))
        legend_style = ParagraphStyle(
            'Legend',
            parent=getSampleStyleSheet()['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#666666')
        )
        legend_text = """
        <b>Notation Guide:</b><br/>
        Lowercase (sa) = Lower octave | Normal (Sa) = Middle octave | UPPERCASE (SA) = Higher octave<br/>
        Keyboard mapping below each line shows corresponding piano keys
        """
        story.append(Paragraph(legend_text, legend_style))
        
        doc.build(story)
        return output_path
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        raise

@app.route('/api/convert', methods=['POST'])
def convert():
    """Main endpoint: YouTube URL to Swaras"""
    try:
        data = request.json
        youtube_url = data.get('youtube_url', '').strip()
        
        if not youtube_url:
            return jsonify({'error': 'YouTube URL required'}), 400
        
        # Create temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract audio
            logger.info(f"Extracting audio from {youtube_url}")
            audio_path = os.path.join(tmpdir, 'audio.wav')
            extract_youtube_audio(youtube_url, audio_path)
            
            # Detect swaras
            logger.info("Detecting melodic line...")
            swaras, sr = detect_melodic_line(audio_path)
            
            if not swaras:
                return jsonify({'error': 'Could not detect melodic line'}), 400
            
            # Clean up swaras - remove very close duplicates
            cleaned_swaras = []
            last_swara = None
            for swara in swaras:
                if last_swara is None or swara['swara'] != last_swara['swara'] or swara['time'] - last_swara['time'] > 0.1:
                    cleaned_swaras.append(swara)
                    last_swara = swara
            
            # Generate PDF
            logger.info("Generating PDF...")
            pdf_path = os.path.join(tmpdir, 'swaras.pdf')
            generate_pdf(cleaned_swaras, pdf_path)
            
            # Read PDF and return
            with open(pdf_path, 'rb') as f:
                pdf_data = f.read()
            
            return {
                'swaras': [s['swara'] for s in cleaned_swaras],
                'count': len(cleaned_swaras),
                'duration': cleaned_swaras[-1]['time'] if cleaned_swaras else 0
            }
    
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/download', methods=['POST'])
def download():
    """Download the PDF"""
    try:
        data = request.json
        youtube_url = data.get('youtube_url', '').strip()
        
        if not youtube_url:
            return jsonify({'error': 'YouTube URL required'}), 400
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract audio
            audio_path = os.path.join(tmpdir, 'audio.wav')
            extract_youtube_audio(youtube_url, audio_path)
            
            # Detect swaras
            swaras, sr = detect_melodic_line(audio_path)
            
            if not swaras:
                return jsonify({'error': 'Could not detect melodic line'}), 400
            
            # Clean up swaras
            cleaned_swaras = []
            last_swara = None
            for swara in swaras:
                if last_swara is None or swara['swara'] != last_swara['swara'] or swara['time'] - last_swara['time'] > 0.1:
                    cleaned_swaras.append(swara)
                    last_swara = swara
            
            # Generate PDF
            pdf_path = os.path.join(tmpdir, 'swaras.pdf')
            generate_pdf(cleaned_swaras, pdf_path)
            
            return send_file(pdf_path, mimetype='application/pdf', as_attachment=True, download_name='swaras.pdf')
    
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=port)
