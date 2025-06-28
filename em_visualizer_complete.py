"""
Professional Electromagnetic Wave Visualizer - Complete University Edition
Comprehensive implementation for university physics education including:
- Plane wave propagation
- Reflection and refraction
- Standing waves
- Wave interference
- Doppler effect
- Polarization states
- Dipole antenna radiation patterns
- Waveguide mode propagation
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO
import base64
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, List
import traceback
from scipy import special

# Configure matplotlib for better appearance
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'lines.linewidth': 2,
    'grid.alpha': 0.3,
    'axes.grid': True,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

app = Flask(__name__)

# Physical constants with precise values
SPEED_OF_LIGHT = 299792458.0  # m/s (exact)
VACUUM_PERMITTIVITY = 8.854187817e-12  # F/m
VACUUM_PERMEABILITY = 1.25663706212e-6  # H/m
VACUUM_IMPEDANCE = 376.730313668  # Ohms

@dataclass
class WaveParameters:
    """Electromagnetic wave parameters with proper units and calculations"""
    frequency: float  # Hz
    amplitude: float  # V/m for E-field
    
    @property
    def wavelength(self) -> float:
        """Calculate wavelength λ = c/f"""
        return SPEED_OF_LIGHT / self.frequency
    
    @property
    def wave_number(self) -> float:
        """Calculate wave number k = 2π/λ"""
        return 2 * np.pi / self.wavelength
    
    @property
    def angular_frequency(self) -> float:
        """Calculate angular frequency ω = 2πf"""
        return 2 * np.pi * self.frequency
    
    @property
    def period(self) -> float:
        """Calculate period T = 1/f"""
        return 1.0 / self.frequency

@dataclass
class MediumProperties:
    """Properties of an electromagnetic medium"""
    name: str
    relative_permittivity: float  # εr (dimensionless)
    relative_permeability: float = 1.0  # μr (dimensionless)
    conductivity: float = 0.0  # σ (S/m)
    
    @property
    def refractive_index(self) -> float:
        """Calculate refractive index n = √(εr × μr)"""
        return np.sqrt(self.relative_permittivity * self.relative_permeability)
    
    @property
    def wave_speed(self) -> float:
        """Calculate phase velocity in medium v = c/n"""
        return SPEED_OF_LIGHT / self.refractive_index
    
    @property
    def impedance(self) -> float:
        """Calculate characteristic impedance Z = Z0 × √(μr/εr)"""
        return VACUUM_IMPEDANCE * np.sqrt(self.relative_permeability / self.relative_permittivity)

# HTML Template with enhanced UI for additional phenomena
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Electromagnetic Wave Visualizer - University Physics</title>
    
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    
    <style>
        :root {
            --primary-color: #2563eb;
            --primary-dark: #1d4ed8;
            --secondary-color: #7c3aed;
            --success-color: #059669;
            --error-color: #dc2626;
            --warning-color: #d97706;
            --bg-primary: #ffffff;
            --bg-secondary: #f9fafb;
            --bg-tertiary: #f3f4f6;
            --text-primary: #111827;
            --text-secondary: #6b7280;
            --border-color: #e5e7eb;
            --shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background-color: var(--bg-secondary);
            color: var(--text-primary);
            line-height: 1.6;
        }
        
        .header {
            background-color: var(--bg-primary);
            border-bottom: 1px solid var(--border-color);
            padding: 1.5rem 0;
            box-shadow: var(--shadow);
        }
        
        .header-content {
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 2rem;
        }
        
        h1 {
            font-size: 2rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 0.25rem;
        }
        
        .subtitle {
            color: var(--text-secondary);
            font-size: 1rem;
        }
        
        .container {
            max-width: 1400px;
            margin: 2rem auto;
            padding: 0 2rem;
            display: grid;
            grid-template-columns: 380px 1fr;
            gap: 2rem;
            align-items: start;
        }
        
        .card {
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 0.5rem;
            padding: 1.5rem;
            box-shadow: var(--shadow);
        }
        
        .card-header {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border-color);
        }
        
        .form-group {
            margin-bottom: 1.25rem;
        }
        
        label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 500;
            color: var(--text-primary);
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.025em;
        }
        
        input[type="number"],
        select {
            width: 100%;
            padding: 0.625rem 0.875rem;
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 0.375rem;
            color: var(--text-primary);
            font-size: 0.875rem;
            transition: all 0.15s ease;
        }
        
        input[type="number"]:focus,
        select:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }
        
        input[type="range"] {
            width: 100%;
            height: 6px;
            background: var(--bg-tertiary);
            border-radius: 3px;
            outline: none;
            -webkit-appearance: none;
            margin: 0.75rem 0;
        }
        
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            background: var(--primary-color);
            border-radius: 50%;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        
        input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.1);
            box-shadow: 0 0 0 8px rgba(37, 99, 235, 0.1);
        }
        
        .range-value {
            display: inline-block;
            background: var(--bg-tertiary);
            color: var(--text-primary);
            padding: 0.25rem 0.75rem;
            border-radius: 0.25rem;
            font-size: 0.875rem;
            font-weight: 500;
            margin-left: 0.5rem;
        }
        
        .btn {
            width: 100%;
            padding: 0.75rem 1.5rem;
            background: var(--primary-color);
            color: white;
            border: none;
            border-radius: 0.375rem;
            font-size: 0.875rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        
        .btn:hover {
            background: var(--primary-dark);
            transform: translateY(-1px);
            box-shadow: var(--shadow-lg);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        .btn-secondary {
            background: var(--bg-tertiary);
            color: var(--text-primary);
            padding: 0.5rem 1rem;
            font-size: 0.875rem;
        }
        
        .btn-secondary:hover {
            background: var(--border-color);
        }
        
        .status {
            margin-top: 1rem;
            padding: 0.75rem 1rem;
            border-radius: 0.375rem;
            display: none;
            font-size: 0.875rem;
        }
        
        .status.success {
            background: #ecfdf5;
            border: 1px solid #6ee7b7;
            color: var(--success-color);
        }
        
        .status.error {
            background: #fef2f2;
            border: 1px solid #fca5a5;
            color: var(--error-color);
        }
        
        #visualization {
            min-height: 400px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        #visualization img {
            max-width: 100%;
            height: auto;
            border-radius: 0.375rem;
            box-shadow: var(--shadow);
        }
        
        .config-section {
            margin-top: 1.5rem;
            border-top: 1px solid var(--border-color);
            padding-top: 1rem;
        }
        
        .config-toggle {
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
            padding: 0.5rem 0;
        }
        
        .config-toggle:hover {
            color: var(--primary-color);
        }
        
        .config-content {
            display: none;
            margin-top: 1rem;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 0.375rem;
            padding: 1rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            overflow-x: auto;
            max-height: 300px;
            overflow-y: auto;
        }
        
        .config-content.show {
            display: block;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 3rem;
            color: var(--text-secondary);
        }
        
        .spinner {
            width: 40px;
            height: 40px;
            margin: 0 auto 1rem;
            border: 3px solid var(--bg-tertiary);
            border-top: 3px solid var(--primary-color);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .viz-description {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 0.375rem;
            padding: 1rem;
            margin-top: 1rem;
            font-size: 0.875rem;
            line-height: 1.6;
        }
        
        .viz-description h3 {
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: var(--text-primary);
        }
        
        .viz-description p {
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }
        
        .viz-description .equation {
            background: var(--bg-primary);
            padding: 0.75rem;
            border-radius: 0.25rem;
            margin: 0.5rem 0;
            text-align: center;
            font-family: 'JetBrains Mono', monospace;
        }
        
        @media (max-width: 1024px) {
            .container {
                grid-template-columns: 1fr;
                gap: 1.5rem;
            }
        }
        
        .text-muted {
            color: var(--text-secondary);
        }
        
        .mt-2 { margin-top: 0.5rem; }
        .mt-4 { margin-top: 1rem; }
        .flex { display: flex; }
        .gap-2 { gap: 0.5rem; }
        .items-center { align-items: center; }
    </style>
</head>
<body>
    <header class="header">
        <div class="header-content">
            <h1>Electromagnetic Wave Visualizer</h1>
            <p class="subtitle">University Physics Education Tool - Complete Edition</p>
        </div>
    </header>
    
    <div class="container">
        <div class="card">
            <h2 class="card-header">Wave Parameters</h2>
            
            <form id="config-form">
                <div class="form-group">
                    <label for="phenomenon">Physical Phenomenon</label>
                    <select id="phenomenon" name="phenomenon" onchange="updatePhenomenon()">
                        <option value="plane_wave">Plane Wave Propagation</option>
                        <option value="standing_wave">Standing Wave Formation</option>
                        <option value="reflection">Reflection and Refraction</option>
                        <option value="interference">Wave Interference</option>
                        <option value="doppler">Doppler Effect</option>
                        <option value="polarization">Polarization</option>
                        <option value="dipole">Dipole Antenna Radiation</option>
                        <option value="waveguide">Waveguide Propagation</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="frequency">Frequency</label>
                    <div class="flex gap-2">
                        <input type="number" id="frequency" name="frequency" value="1" step="0.1" min="0.001" style="flex: 1;">
                        <select id="freq_unit" name="freq_unit" style="width: 80px;">
                            <option value="1e9">GHz</option>
                            <option value="1e6">MHz</option>
                            <option value="1e3">kHz</option>
                            <option value="1">Hz</option>
                        </select>
                    </div>
                    <p class="text-muted mt-2" style="font-size: 0.75rem;">
                        Wavelength: <span id="wavelength-display">0.300 m</span>
                    </p>
                </div>
                
                <div class="form-group">
                    <label for="amplitude">Electric Field Amplitude (V/m)</label>
                    <input type="number" id="amplitude" name="amplitude" value="1" step="0.1" min="0.1">
                </div>
                
                <div id="additional-params"></div>
                
                <button type="button" class="btn" onclick="generateVisualization()">
                    Generate Visualization
                </button>
                
                <div id="status" class="status"></div>
                
                <div class="config-section">
                    <div class="config-toggle" onclick="toggleConfig()">
                        <span>View Configuration</span>
                        <span id="config-arrow">▶</span>
                    </div>
                    <pre class="config-content" id="config-content"></pre>
                    <button class="btn-secondary mt-2" onclick="downloadConfig()" style="display: none;" id="download-btn">
                        Download Configuration
                    </button>
                </div>
            </form>
        </div>
        
        <div class="card">
            <h2 class="card-header">Visualization Results</h2>
            
            <div id="visualization">
                <div style="text-align: center; color: var(--text-secondary);">
                    <p>Select parameters and click "Generate Visualization" to begin.</p>
                    <p style="font-size: 0.875rem; margin-top: 0.5rem;">
                        This tool provides accurate electromagnetic wave simulations for educational purposes.
                    </p>
                </div>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Generating visualization...</p>
            </div>
            
            <div class="viz-description" id="viz-description" style="display: none;"></div>
        </div>
    </div>
    
    <script>
        function updateWavelength() {
            const freq = parseFloat(document.getElementById('frequency').value);
            const unit = parseFloat(document.getElementById('freq_unit').value);
            const frequency = freq * unit;
            
            const c = 299792458;
            const wavelength = c / frequency;
            
            let display;
            if (wavelength >= 1) {
                display = wavelength.toFixed(3) + ' m';
            } else if (wavelength >= 0.001) {
                display = (wavelength * 1000).toFixed(3) + ' mm';
            } else if (wavelength >= 0.000001) {
                display = (wavelength * 1e6).toFixed(3) + ' μm';
            } else {
                display = (wavelength * 1e9).toFixed(3) + ' nm';
            }
            
            document.getElementById('wavelength-display').textContent = display;
        }
        
        document.getElementById('frequency').addEventListener('input', updateWavelength);
        document.getElementById('freq_unit').addEventListener('change', updateWavelength);
        
        function updatePhenomenon() {
            const phenomenon = document.getElementById('phenomenon').value;
            const additionalParams = document.getElementById('additional-params');
            
            let html = '';
            
            if (phenomenon === 'reflection') {
                html = `
                    <div class="form-group">
                        <label for="medium1">Incident Medium</label>
                        <select id="medium1" name="medium1">
                            <option value="Air">Air (n = 1.000)</option>
                            <option value="Water">Water (n = 1.333)</option>
                            <option value="Glass">Glass (n = 1.500)</option>
                            <option value="Diamond">Diamond (n = 2.417)</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="medium2">Transmitted Medium</label>
                        <select id="medium2" name="medium2">
                            <option value="Glass">Glass (n = 1.500)</option>
                            <option value="Water">Water (n = 1.333)</option>
                            <option value="Air">Air (n = 1.000)</option>
                            <option value="Diamond">Diamond (n = 2.417)</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="angle">
                            Incident Angle
                            <span class="range-value" id="angle-value">30°</span>
                        </label>
                        <input type="range" id="angle" name="angle" min="0" max="90" value="30" step="1"
                               oninput="document.getElementById('angle-value').textContent = this.value + '°'">
                    </div>
                `;
            } else if (phenomenon === 'interference') {
                html = `
                    <div class="form-group">
                        <label for="sources">
                            Number of Sources
                            <span class="range-value" id="sources-value">2</span>
                        </label>
                        <input type="range" id="sources" name="sources" min="2" max="5" value="2" step="1"
                               oninput="document.getElementById('sources-value').textContent = this.value">
                    </div>
                    <div class="form-group">
                        <label for="separation">
                            Source Separation
                            <span class="range-value" id="sep-value">2.0λ</span>
                        </label>
                        <input type="range" id="separation" name="separation" min="0.5" max="5" value="2" step="0.5"
                               oninput="document.getElementById('sep-value').textContent = this.value + 'λ'">
                    </div>
                `;
            } else if (phenomenon === 'doppler') {
                html = `
                    <div class="form-group">
                        <label for="velocity">
                            Source Velocity
                            <span class="range-value" id="vel-value">0 m/s</span>
                        </label>
                        <input type="range" id="velocity" name="velocity" min="-100000" max="100000" value="0" step="1000"
                               oninput="document.getElementById('vel-value').textContent = this.value + ' m/s'">
                        <p class="text-muted mt-2" style="font-size: 0.75rem;">
                            Positive: approaching observer, Negative: receding
                        </p>
                    </div>
                `;
            } else if (phenomenon === 'polarization') {
                html = `
                    <div class="form-group">
                        <label for="pol_type">Polarization Type</label>
                        <select id="pol_type" name="pol_type">
                            <option value="linear">Linear Polarization</option>
                            <option value="circular">Circular Polarization</option>
                            <option value="elliptical">Elliptical Polarization</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="pol_angle">
                            Polarization Angle
                            <span class="range-value" id="pol-value">0°</span>
                        </label>
                        <input type="range" id="pol_angle" name="pol_angle" min="0" max="180" value="0" step="5"
                               oninput="document.getElementById('pol-value').textContent = this.value + '°'">
                    </div>
                `;
            } else if (phenomenon === 'dipole') {
                html = `
                    <div class="form-group">
                        <label for="dipole_length">
                            Dipole Length
                            <span class="range-value" id="dipole-value">0.5λ</span>
                        </label>
                        <input type="range" id="dipole_length" name="dipole_length" min="0.1" max="2.0" value="0.5" step="0.1"
                               oninput="document.getElementById('dipole-value').textContent = this.value + 'λ'">
                    </div>
                    <div class="form-group">
                        <label for="current_dist">Current Distribution</label>
                        <select id="current_dist" name="current_dist">
                            <option value="uniform">Uniform</option>
                            <option value="sinusoidal">Sinusoidal</option>
                            <option value="triangular">Triangular</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="view_type">View Type</label>
                        <select id="view_type" name="view_type">
                            <option value="3d">3D Pattern</option>
                            <option value="2d_elevation">2D E-plane (Elevation)</option>
                            <option value="2d_azimuth">2D H-plane (Azimuth)</option>
                        </select>
                    </div>
                `;
            } else if (phenomenon === 'waveguide') {
                html = `
                    <div class="form-group">
                        <label for="guide_type">Waveguide Type</label>
                        <select id="guide_type" name="guide_type">
                            <option value="rectangular">Rectangular</option>
                            <option value="circular">Circular</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="guide_width">
                            Guide Width (a)
                            <span class="range-value" id="width-value">23.0 mm</span>
                        </label>
                        <input type="range" id="guide_width" name="guide_width" min="10" max="100" value="23" step="1"
                               oninput="document.getElementById('width-value').textContent = this.value + ' mm'">
                    </div>
                    <div class="form-group">
                        <label for="guide_height">
                            Guide Height (b)
                            <span class="range-value" id="height-value">10.0 mm</span>
                        </label>
                        <input type="range" id="guide_height" name="guide_height" min="5" max="50" value="10" step="1"
                               oninput="document.getElementById('height-value').textContent = this.value + ' mm'">
                    </div>
                    <div class="form-group">
                        <label for="mode">Propagation Mode</label>
                        <select id="mode" name="mode">
                            <option value="TE10">TE10 (Dominant)</option>
                            <option value="TE20">TE20</option>
                            <option value="TE01">TE01</option>
                            <option value="TE11">TE11</option>
                            <option value="TM11">TM11</option>
                        </select>
                    </div>
                `;
            }
            
            additionalParams.innerHTML = html;
            updateWavelength();
        }
        
        function toggleConfig() {
            const content = document.getElementById('config-content');
            const arrow = document.getElementById('config-arrow');
            const downloadBtn = document.getElementById('download-btn');
            
            content.classList.toggle('show');
            arrow.textContent = content.classList.contains('show') ? '▼' : '▶';
            downloadBtn.style.display = content.classList.contains('show') ? 'block' : 'none';
        }
        
        function showStatus(message, type) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = 'status ' + type;
            status.style.display = 'block';
            
            setTimeout(() => {
                status.style.display = 'none';
            }, 5000);
        }
        
        function generateVisualization() {
            const formData = {
                phenomenon: document.getElementById('phenomenon').value,
                frequency: parseFloat(document.getElementById('frequency').value) * 
                           parseFloat(document.getElementById('freq_unit').value),
                amplitude: parseFloat(document.getElementById('amplitude').value)
            };
            
            if (formData.phenomenon === 'reflection') {
                formData.medium1 = document.getElementById('medium1').value;
                formData.medium2 = document.getElementById('medium2').value;
                formData.angle = parseFloat(document.getElementById('angle').value);
            } else if (formData.phenomenon === 'interference') {
                formData.sources = parseInt(document.getElementById('sources').value);
                formData.separation = parseFloat(document.getElementById('separation').value);
            } else if (formData.phenomenon === 'doppler') {
                formData.velocity = parseFloat(document.getElementById('velocity').value);
            } else if (formData.phenomenon === 'polarization') {
                formData.pol_type = document.getElementById('pol_type').value;
                formData.pol_angle = parseFloat(document.getElementById('pol_angle').value);
            } else if (formData.phenomenon === 'dipole') {
                formData.dipole_length = parseFloat(document.getElementById('dipole_length').value);
                formData.current_dist = document.getElementById('current_dist').value;
                formData.view_type = document.getElementById('view_type').value;
            } else if (formData.phenomenon === 'waveguide') {
                formData.guide_type = document.getElementById('guide_type').value;
                formData.guide_width = parseFloat(document.getElementById('guide_width').value) / 1000; // Convert to m
                formData.guide_height = parseFloat(document.getElementById('guide_height').value) / 1000; // Convert to m
                formData.mode = document.getElementById('mode').value;
            }
            
            document.getElementById('visualization').style.display = 'none';
            document.getElementById('loading').style.display = 'block';
            document.getElementById('status').style.display = 'none';
            document.getElementById('viz-description').style.display = 'none';
            
            fetch('/visualize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('visualization').style.display = 'flex';
                
                if (data.success) {
                    document.getElementById('visualization').innerHTML = 
                        `<img src="data:image/png;base64,${data.image}" alt="${data.description}">`;
                    
                    document.getElementById('config-content').textContent = 
                        JSON.stringify(data.config, null, 2);
                    
                    document.getElementById('viz-description').innerHTML = data.description_html;
                    document.getElementById('viz-description').style.display = 'block';
                    
                    showStatus('Visualization generated successfully', 'success');
                    
                    if (window.MathJax) {
                        MathJax.typesetPromise();
                    }
                } else {
                    showStatus('Error: ' + data.error, 'error');
                    document.getElementById('visualization').innerHTML = 
                        '<p style="color: var(--error-color);">Failed to generate visualization</p>';
                }
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('visualization').style.display = 'flex';
                showStatus('Network error: ' + error, 'error');
            });
        }
        
        function downloadConfig() {
            const config = document.getElementById('config-content').textContent;
            const blob = new Blob([config], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
            a.download = `em_wave_config_${timestamp}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            showStatus('Configuration downloaded', 'success');
        }
        
        document.addEventListener('DOMContentLoaded', function() {
            updatePhenomenon();
            updateWavelength();
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serve the HTML interface"""
    return HTML_TEMPLATE

@app.route('/visualize', methods=['POST'])
def visualize():
    """Handle visualization requests with error handling"""
    try:
        data = request.json
        
        # Validate input data
        if data['frequency'] <= 0:
            raise ValueError("Frequency must be positive")
        if data['amplitude'] <= 0:
            raise ValueError("Amplitude must be positive")
        
        # Generate configuration
        config = generate_configuration(data)
        
        # Create visualization based on phenomenon
        if data['phenomenon'] == 'plane_wave':
            result = create_plane_wave_visualization(config)
        elif data['phenomenon'] == 'standing_wave':
            result = create_standing_wave_visualization(config)
        elif data['phenomenon'] == 'reflection':
            result = create_reflection_visualization(config, data)
        elif data['phenomenon'] == 'interference':
            result = create_interference_visualization(config, data)
        elif data['phenomenon'] == 'doppler':
            result = create_doppler_visualization(config, data)
        elif data['phenomenon'] == 'polarization':
            result = create_polarization_visualization(config, data)
        elif data['phenomenon'] == 'dipole':
            result = create_dipole_visualization(config, data)
        elif data['phenomenon'] == 'waveguide':
            result = create_waveguide_visualization(config, data)
        else:
            raise ValueError(f"Unknown phenomenon: {data['phenomenon']}")
        
        return jsonify({
            'success': True,
            'image': result['image'],
            'config': config,
            'description': result.get('description', ''),
            'description_html': result.get('description_html', '')
        })
        
    except Exception as e:
        # Log the full traceback for debugging
        print(f"Error in visualization: {str(e)}")
        print(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': str(e)
        })

def generate_configuration(data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate detailed configuration with physics parameters"""
    wave = WaveParameters(
        frequency=data['frequency'],
        amplitude=data['amplitude']
    )
    
    config = {
        "version": "1.0",
        "phenomenon": data['phenomenon'],
        "physics": {
            "wave": {
                "frequency": wave.frequency,
                "amplitude": wave.amplitude,
                "wavelength": wave.wavelength,
                "period": wave.period,
                "angular_frequency": wave.angular_frequency,
                "wave_number": wave.wave_number
            },
            "constants": {
                "speed_of_light": SPEED_OF_LIGHT,
                "vacuum_permittivity": VACUUM_PERMITTIVITY,
                "vacuum_permeability": VACUUM_PERMEABILITY,
                "vacuum_impedance": VACUUM_IMPEDANCE
            }
        }
    }
    
    # Add phenomenon-specific parameters
    if data['phenomenon'] == 'reflection':
        # Define materials with correct properties
        materials = {
            "Air": MediumProperties("Air", 1.00059),
            "Water": MediumProperties("Water", 80.0),
            "Glass": MediumProperties("Glass", 2.25),
            "Diamond": MediumProperties("Diamond", 5.84)
        }
        
        medium1 = materials[data['medium1']]
        medium2 = materials[data['medium2']]
        
        config["physics"]["media"] = [
            {
                "name": medium1.name,
                "relative_permittivity": medium1.relative_permittivity,
                "relative_permeability": medium1.relative_permeability,
                "refractive_index": medium1.refractive_index,
                "wave_speed": medium1.wave_speed,
                "impedance": medium1.impedance
            },
            {
                "name": medium2.name,
                "relative_permittivity": medium2.relative_permittivity,
                "relative_permeability": medium2.relative_permeability,
                "refractive_index": medium2.refractive_index,
                "wave_speed": medium2.wave_speed,
                "impedance": medium2.impedance
            }
        ]
        
        # Calculate reflection and transmission parameters
        theta_i = np.radians(data['angle'])
        n1 = medium1.refractive_index
        n2 = medium2.refractive_index
        
        config["physics"]["incident_angle_deg"] = data['angle']
        config["physics"]["incident_angle_rad"] = theta_i
        
        # Apply Snell's law
        sin_theta_t = n1 * np.sin(theta_i) / n2
        
        if sin_theta_t <= 1.0:
            # Normal refraction
            theta_t = np.arcsin(sin_theta_t)
            config["physics"]["transmitted_angle_deg"] = np.degrees(theta_t)
            config["physics"]["transmitted_angle_rad"] = theta_t
            config["physics"]["total_internal_reflection"] = False
            
            # Calculate Fresnel coefficients
            cos_theta_i = np.cos(theta_i)
            cos_theta_t = np.cos(theta_t)
            
            # TE (s-polarized) coefficients
            r_te = (n1 * cos_theta_i - n2 * cos_theta_t) / (n1 * cos_theta_i + n2 * cos_theta_t)
            t_te = (2 * n1 * cos_theta_i) / (n1 * cos_theta_i + n2 * cos_theta_t)
            
            # TM (p-polarized) coefficients
            r_tm = (n2 * cos_theta_i - n1 * cos_theta_t) / (n2 * cos_theta_i + n1 * cos_theta_t)
            t_tm = (2 * n1 * cos_theta_i) / (n2 * cos_theta_i + n1 * cos_theta_t)
            
            # Power coefficients
            R_te = r_te**2
            T_te = (n2 * cos_theta_t) / (n1 * cos_theta_i) * t_te**2
            
            config["physics"]["fresnel_coefficients"] = {
                "r_te": float(r_te),
                "t_te": float(t_te),
                "r_tm": float(r_tm),
                "t_tm": float(t_tm),
                "R_te": float(R_te),
                "T_te": float(T_te)
            }
        else:
            # Total internal reflection
            config["physics"]["total_internal_reflection"] = True
            config["physics"]["critical_angle_deg"] = np.degrees(np.arcsin(n2/n1))
            config["physics"]["fresnel_coefficients"] = {
                "r_te": 1.0,
                "r_tm": 1.0,
                "R_te": 1.0,
                "R_tm": 1.0
            }
    
    elif data['phenomenon'] == 'doppler':
        v = data.get('velocity', 0)
        
        # Relativistic Doppler effect
        beta = v / SPEED_OF_LIGHT
        
        if abs(beta) < 1.0:
            # Calculate observed frequency
            if v > 0:  # Approaching
                f_observed = wave.frequency * np.sqrt((1 + beta) / (1 - beta))
            else:  # Receding
                f_observed = wave.frequency * np.sqrt((1 - abs(beta)) / (1 + abs(beta)))
            
            config["physics"]["doppler"] = {
                "source_velocity": v,
                "beta": beta,
                "observed_frequency": f_observed,
                "frequency_shift": f_observed - wave.frequency,
                "relative_shift": (f_observed - wave.frequency) / wave.frequency
            }
        else:
            config["physics"]["doppler"] = {
                "source_velocity": v,
                "error": "Velocity cannot exceed speed of light"
            }
    
    elif data['phenomenon'] == 'dipole':
        dipole_length = data.get('dipole_length', 0.5) * wave.wavelength
        config["physics"]["dipole"] = {
            "length": dipole_length,
            "length_wavelengths": data.get('dipole_length', 0.5),
            "current_distribution": data.get('current_dist', 'uniform'),
            "radiation_resistance": calculate_radiation_resistance(dipole_length, wave.wavelength)
        }
    
    elif data['phenomenon'] == 'waveguide':
        a = data.get('guide_width', 0.023)  # m
        b = data.get('guide_height', 0.010)  # m
        mode = data.get('mode', 'TE10')
        
        # Extract mode indices
        mode_type = mode[:2]  # TE or TM
        m = int(mode[2])
        n = int(mode[3])
        
        # Calculate cutoff frequency
        fc = SPEED_OF_LIGHT / (2 * np.pi) * np.sqrt((m * np.pi / a)**2 + (n * np.pi / b)**2)
        
        # Check if frequency is above cutoff
        if wave.frequency > fc:
            # Propagating mode
            beta_g = np.sqrt((wave.angular_frequency / SPEED_OF_LIGHT)**2 - (2 * np.pi * fc / SPEED_OF_LIGHT)**2)
            lambda_g = 2 * np.pi / beta_g
            vp = wave.angular_frequency / beta_g
            vg = SPEED_OF_LIGHT**2 / vp
            
            config["physics"]["waveguide"] = {
                "type": data.get('guide_type', 'rectangular'),
                "width": a,
                "height": b,
                "mode": mode,
                "mode_type": mode_type,
                "mode_indices": {"m": m, "n": n},
                "cutoff_frequency": fc,
                "cutoff_wavelength": SPEED_OF_LIGHT / fc,
                "propagation_constant": beta_g,
                "guide_wavelength": lambda_g,
                "phase_velocity": vp,
                "group_velocity": vg,
                "impedance": calculate_waveguide_impedance(mode_type, wave.frequency, fc)
            }
        else:
            # Evanescent mode
            alpha = np.sqrt((2 * np.pi * fc / SPEED_OF_LIGHT)**2 - (wave.angular_frequency / SPEED_OF_LIGHT)**2)
            config["physics"]["waveguide"] = {
                "type": data.get('guide_type', 'rectangular'),
                "width": a,
                "height": b,
                "mode": mode,
                "cutoff_frequency": fc,
                "attenuation_constant": alpha,
                "evanescent": True
            }
    
    return config

def calculate_radiation_resistance(dipole_length: float, wavelength: float) -> float:
    """Calculate radiation resistance of a dipole antenna"""
    l_over_lambda = dipole_length / wavelength
    
    if l_over_lambda <= 0.1:
        # Short dipole approximation
        R_rad = 20 * (np.pi * l_over_lambda)**2
    elif abs(l_over_lambda - 0.5) < 0.01:
        # Half-wave dipole
        R_rad = 73.13
    else:
        # General formula (approximate)
        R_rad = 80 * (np.pi * l_over_lambda)**2
    
    return R_rad

def calculate_waveguide_impedance(mode_type: str, frequency: float, fc: float) -> float:
    """Calculate characteristic impedance of waveguide mode"""
    if frequency <= fc:
        return np.inf
    
    if mode_type == "TE":
        # TE modes
        return VACUUM_IMPEDANCE / np.sqrt(1 - (fc / frequency)**2)
    else:
        # TM modes
        return VACUUM_IMPEDANCE * np.sqrt(1 - (fc / frequency)**2)

def create_plane_wave_visualization(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create plane wave visualization with E and B fields"""
    wave = config["physics"]["wave"]
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle('Electromagnetic Plane Wave Propagation', fontsize=14, fontweight='bold')
    
    # Spatial parameters - show 3 wavelengths
    x = np.linspace(0, 3 * wave["wavelength"], 1000)
    
    # Calculate fields at t=0
    k = wave["wave_number"]
    E = wave["amplitude"] * np.sin(k * x)
    
    # Magnetic field: B = E/c for plane waves
    B = E / SPEED_OF_LIGHT
    
    # Plot E-field
    ax1.plot(x * 1e3, E, 'b-', linewidth=2.5, label='E-field')
    ax1.fill_between(x * 1e3, 0, E, alpha=0.3, color='blue')
    ax1.set_ylabel('Electric Field (V/m)', fontsize=11)
    ax1.set_title('Electric Field Component', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_ylim(-1.5 * wave["amplitude"], 1.5 * wave["amplitude"])
    
    # Add wavelength annotation
    for i in range(3):
        x_start = i * wave["wavelength"] * 1e3
        x_end = (i + 1) * wave["wavelength"] * 1e3
        y_pos = 0.8 * wave["amplitude"]
        
        if i == 0:  # Only annotate the first wavelength
            ax1.annotate('', xy=(x_end, y_pos), xytext=(x_start, y_pos),
                        arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
            ax1.text((x_start + x_end) / 2, y_pos * 1.1, 
                    f'λ = {wave["wavelength"]*1e3:.1f} mm',
                    ha='center', va='bottom', fontsize=10)
    
    # Plot B-field
    ax2.plot(x * 1e3, B * 1e9, 'r-', linewidth=2.5, label='B-field')
    ax2.fill_between(x * 1e3, 0, B * 1e9, alpha=0.3, color='red')
    ax2.set_ylabel('Magnetic Field (nT)', fontsize=11)
    ax2.set_title('Magnetic Field Component', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.set_ylim(-1.5 * wave["amplitude"] / SPEED_OF_LIGHT * 1e9, 
                  1.5 * wave["amplitude"] / SPEED_OF_LIGHT * 1e9)
    
    # Plot energy density
    u_E = 0.5 * VACUUM_PERMITTIVITY * E**2
    u_B = 0.5 * B**2 / VACUUM_PERMEABILITY
    u_total = u_E + u_B
    
    # Convert to more readable units (μJ/m³)
    ax3.plot(x * 1e3, u_E * 1e6, 'b--', linewidth=2, label='Electric energy', alpha=0.7)
    ax3.plot(x * 1e3, u_B * 1e6, 'r--', linewidth=2, label='Magnetic energy', alpha=0.7)
    ax3.plot(x * 1e3, u_total * 1e6, 'k-', linewidth=2.5, label='Total energy')
    ax3.set_xlabel('Position (mm)', fontsize=11)
    ax3.set_ylabel('Energy Density (μJ/m³)', fontsize=11)
    ax3.set_title('Electromagnetic Energy Density', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')
    ax3.set_ylim(0, max(u_total) * 1.2 * 1e6)
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    
    # Create description
    description_html = f"""
    <h3>Plane Electromagnetic Wave</h3>
    <p>This visualization shows a plane electromagnetic wave propagating in free space. The electric and magnetic fields oscillate perpendicular to each other and to the direction of propagation.</p>
    
    <div class="equation">
        E(x,t) = E₀ sin(kx - ωt)<br>
        B(x,t) = E₀/c sin(kx - ωt)
    </div>
    
    <p><strong>Key Parameters:</strong></p>
    <ul>
        <li>Frequency: {wave['frequency']/1e9:.3f} GHz</li>
        <li>Wavelength: {wave['wavelength']*1e3:.3f} mm</li>
        <li>Wave number: k = {wave['wave_number']:.3f} rad/m</li>
        <li>Period: {wave['period']*1e9:.3f} ns</li>
        <li>B-field amplitude: {wave['amplitude']/SPEED_OF_LIGHT*1e9:.3f} nT</li>
    </ul>
    
    <p>Note that the electric and magnetic energy densities are equal in a plane wave, demonstrating the equipartition of energy in electromagnetic radiation.</p>
    """
    
    return {
        'image': image_base64,
        'description_html': description_html
    }

def create_reflection_visualization(config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Create reflection and refraction visualization with correct physics"""
    wave = config["physics"]["wave"]
    media = config["physics"]["media"]
    theta_i = config["physics"]["incident_angle_rad"]
    
    n1 = media[0]["refractive_index"]
    n2 = media[1]["refractive_index"]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[2, 2, 1], height_ratios=[1, 1])
    ax_field = fig.add_subplot(gs[:, 0])
    ax_ray = fig.add_subplot(gs[0, 1])
    ax_fresnel = fig.add_subplot(gs[1, 1])
    ax_info = fig.add_subplot(gs[:, 2])
    
    fig.suptitle('Electromagnetic Wave Reflection and Refraction', fontsize=14, fontweight='bold')
    
    # Field visualization
    x = np.linspace(-3, 3, 300)
    y = np.linspace(-3, 3, 300)
    X, Y = np.meshgrid(x, y)
    
    # Wave parameters
    k = 2 * np.pi / wave["wavelength"]
    k1 = k * n1
    k2 = k * n2
    
    # Initialize field
    E_total = np.zeros_like(X)
    
    if not config["physics"]["total_internal_reflection"]:
        # Normal refraction
        theta_t = config["physics"]["transmitted_angle_rad"]
        
        # Wave vectors
        k_inc_x = k1 * np.sin(theta_i)
        k_inc_y = -k1 * np.cos(theta_i)
        
        k_ref_x = k1 * np.sin(theta_i)
        k_ref_y = k1 * np.cos(theta_i)
        
        k_trans_x = k2 * np.sin(theta_t)
        k_trans_y = -k2 * np.cos(theta_t)
        
        # Fields
        E_inc = wave["amplitude"] * np.sin(k_inc_x * X + k_inc_y * Y)
        E_ref = config["physics"]["fresnel_coefficients"]["r_te"] * wave["amplitude"] * \
                np.sin(k_ref_x * X + k_ref_y * Y)
        E_trans = config["physics"]["fresnel_coefficients"]["t_te"] * wave["amplitude"] * \
                  np.sin(k_trans_x * X + k_trans_y * Y)
        
        # Combine fields
        E_total[Y >= 0] = E_inc[Y >= 0] + E_ref[Y >= 0]
        E_total[Y < 0] = E_trans[Y < 0]
    else:
        # Total internal reflection
        k_inc_x = k1 * np.sin(theta_i)
        k_inc_y = -k1 * np.cos(theta_i)
        
        k_ref_x = k1 * np.sin(theta_i)
        k_ref_y = k1 * np.cos(theta_i)
        
        E_inc = wave["amplitude"] * np.sin(k_inc_x * X + k_inc_y * Y)
        E_ref = wave["amplitude"] * np.sin(k_ref_x * X + k_ref_y * Y)
        
        E_total[Y >= 0] = E_inc[Y >= 0] + E_ref[Y >= 0]
        
        # Evanescent wave
        alpha = k1 * np.sqrt((n1 * np.sin(theta_i))**2 / n2**2 - 1)
        E_total[Y < 0] = wave["amplitude"] * np.exp(alpha * Y[Y < 0]) * np.sin(k_inc_x * X[Y < 0])
    
    # Plot field
    im = ax_field.contourf(X, Y, E_total, levels=50, cmap='RdBu_r', 
                          vmin=-2*wave["amplitude"], vmax=2*wave["amplitude"])
    ax_field.axhline(y=0, color='black', linewidth=2)
    ax_field.set_xlabel('x (m)', fontsize=11)
    ax_field.set_ylabel('y (m)', fontsize=11)
    ax_field.set_title('Electric Field Distribution', fontsize=12)
    ax_field.set_aspect('equal')
    
    # Add medium labels
    ax_field.text(-2.5, 1.5, f'{media[0]["name"]}\nn = {n1:.3f}', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax_field.text(-2.5, -1.5, f'{media[1]["name"]}\nn = {n2:.3f}', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Ray diagram
    ax_ray.set_xlim(-2, 2)
    ax_ray.set_ylim(-2, 2)
    ax_ray.axhline(y=0, color='black', linewidth=2)
    ax_ray.axvline(x=0, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    
    # Draw rays with correct arrow function
    # Incident ray
    inc_start_x = -1.5 * np.sin(theta_i)
    inc_start_y = 1.5 * np.cos(theta_i)
    inc_dx = 1.4 * np.sin(theta_i)
    inc_dy = -1.4 * np.cos(theta_i)
    
    ax_ray.arrow(inc_start_x, inc_start_y, inc_dx, inc_dy,
                head_width=0.1, head_length=0.08, fc='blue', ec='blue', linewidth=2)
    ax_ray.text(inc_start_x - 0.2, inc_start_y + 0.2, 'Incident',
               ha='center', color='blue', fontweight='bold')
    
    # Reflected ray
    ref_dx = 1.4 * np.sin(theta_i)
    ref_dy = 1.4 * np.cos(theta_i)
    
    ax_ray.arrow(0, 0, ref_dx, ref_dy,
                head_width=0.1, head_length=0.08, fc='red', ec='red', linewidth=2)
    ax_ray.text(1.6 * np.sin(theta_i), 1.6 * np.cos(theta_i), 'Reflected',
               ha='center', color='red', fontweight='bold')
    
    # Transmitted ray (if not TIR)
    if not config["physics"]["total_internal_reflection"]:
        theta_t = config["physics"]["transmitted_angle_rad"]
        trans_dx = 1.4 * np.sin(theta_t)
        trans_dy = -1.4 * np.cos(theta_t)
        
        ax_ray.arrow(0, 0, trans_dx, trans_dy,
                    head_width=0.1, head_length=0.08, fc='green', ec='green', linewidth=2)
        ax_ray.text(1.6 * np.sin(theta_t), -1.6 * np.cos(theta_t), 'Transmitted',
                   ha='center', color='green', fontweight='bold')
    else:
        ax_ray.text(0, -1, 'Total Internal\nReflection', ha='center',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                   fontsize=11, fontweight='bold')
    
    ax_ray.set_xlabel('x', fontsize=11)
    ax_ray.set_ylabel('y', fontsize=11)
    ax_ray.set_title('Ray Diagram', fontsize=12)
    ax_ray.grid(True, alpha=0.3)
    ax_ray.set_aspect('equal')
    
    # Fresnel coefficients plot
    angles = np.linspace(0, 90, 200)
    angles_rad = np.radians(angles)
    
    R_te_list = []
    R_tm_list = []
    
    for angle in angles_rad:
        sin_t = n1 * np.sin(angle) / n2
        if sin_t <= 1:
            cos_i = np.cos(angle)
            cos_t = np.cos(np.arcsin(sin_t))
            
            # TE polarization
            r_te = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
            R_te_list.append(r_te**2)
            
            # TM polarization
            r_tm = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)
            R_tm_list.append(r_tm**2)
        else:
            R_te_list.append(1.0)
            R_tm_list.append(1.0)
    
    ax_fresnel.plot(angles, R_te_list, 'b-', linewidth=2, label='TE (s-pol)')
    ax_fresnel.plot(angles, R_tm_list, 'r-', linewidth=2, label='TM (p-pol)')
    ax_fresnel.axvline(x=np.degrees(theta_i), color='green', linestyle='--', alpha=0.7, linewidth=2)
    ax_fresnel.set_xlabel('Incident Angle (degrees)', fontsize=11)
    ax_fresnel.set_ylabel('Reflectance', fontsize=11)
    ax_fresnel.set_title('Fresnel Reflectance', fontsize=12)
    ax_fresnel.grid(True, alpha=0.3)
    ax_fresnel.legend()
    ax_fresnel.set_xlim(0, 90)
    ax_fresnel.set_ylim(0, 1.05)
    
    # Information panel
    ax_info.axis('off')
    
    if not config["physics"]["total_internal_reflection"]:
        info_text = f"""Analysis Results
        
Incident angle: {np.degrees(theta_i):.1f}°
Transmitted angle: {np.degrees(config["physics"]["transmitted_angle_rad"]):.1f}°

Fresnel Coefficients (TE):
r = {config["physics"]["fresnel_coefficients"]["r_te"]:.3f}
t = {config["physics"]["fresnel_coefficients"]["t_te"]:.3f}

Power Reflectance:
R = {config["physics"]["fresnel_coefficients"]["R_te"]:.3f}
T = {config["physics"]["fresnel_coefficients"]["T_te"]:.3f}

Energy Conservation:
R + T = {config["physics"]["fresnel_coefficients"]["R_te"] + 
         config["physics"]["fresnel_coefficients"]["T_te"]:.3f}"""
    else:
        info_text = f"""Total Internal Reflection
        
Incident angle: {np.degrees(theta_i):.1f}°
Critical angle: {config["physics"]["critical_angle_deg"]:.1f}°

The incident angle exceeds the
critical angle, resulting in
total internal reflection.

All incident power is reflected
back into the first medium.

An evanescent wave exists in
the second medium with decay
length ~ λ/{2*np.pi:.1f}"""
    
    ax_info.text(0.1, 0.9, info_text, transform=ax_info.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    
    # Create description
    description_html = f"""
    <h3>Reflection and Refraction at a Dielectric Interface</h3>
    <p>This visualization shows electromagnetic wave behavior at the boundary between two dielectric media, governed by Snell's law and the Fresnel equations.</p>
    
    <div class="equation">
        Snell's Law: n₁ sin θ₁ = n₂ sin θ₂<br>
        Fresnel (TE): r = (n₁cos θ₁ - n₂cos θ₂)/(n₁cos θ₁ + n₂cos θ₂)
    </div>
    
    <p><strong>Physical Principles:</strong></p>
    <ul>
        <li>Boundary conditions require continuity of tangential E and H fields</li>
        <li>Power conservation: R + T = 1 for lossless media</li>
        <li>Phase matching at the interface determines ray directions</li>
        <li>Total internal reflection occurs when sin θ₁ > n₂/n₁</li>
    </ul>
    """
    
    return {
        'image': image_base64,
        'description_html': description_html
    }

def create_standing_wave_visualization(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create standing wave visualization"""
    wave = config["physics"]["wave"]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('Standing Wave Formation', fontsize=14, fontweight='bold')
    
    # Create standing wave in a cavity
    L = 2.5 * wave["wavelength"]  # Cavity length
    x = np.linspace(0, L, 1000)
    
    # Standing wave pattern
    k = wave["wave_number"]
    
    # Show multiple time snapshots
    times = np.linspace(0, wave["period"], 9)
    colors = plt.cm.viridis(np.linspace(0, 1, len(times)))
    
    for i, t in enumerate(times):
        phase = wave["angular_frequency"] * t
        E_standing = 2 * wave["amplitude"] * np.cos(k * x) * np.cos(phase)
        ax1.plot(x * 1e3, E_standing, color=colors[i], alpha=0.7, linewidth=2)
    
    # Plot envelope
    envelope = 2 * wave["amplitude"] * np.abs(np.cos(k * x))
    ax1.plot(x * 1e3, envelope, 'k--', linewidth=2.5, label='Envelope')
    ax1.plot(x * 1e3, -envelope, 'k--', linewidth=2.5)
    
    # Mark nodes
    node_positions = []
    for i in range(int(L / (wave["wavelength"] / 2)) + 1):
        node_x = (i * wave["wavelength"] / 2)
        if node_x <= L and i % 2 == 1:  # Only odd multiples for cos pattern
            node_positions.append(node_x)
            ax1.axvline(x=node_x * 1e3, color='red', alpha=0.3, linestyle=':')
            ax1.plot(node_x * 1e3, 0, 'ro', markersize=8)
    
    ax1.set_ylabel('Electric Field (V/m)', fontsize=11)
    ax1.set_title('Standing Wave Pattern (Multiple Time Steps)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(-2.5 * wave["amplitude"], 2.5 * wave["amplitude"])
    
    # Energy distribution
    # Average energy density over one period
    energy_density = (VACUUM_PERMITTIVITY / 2) * envelope**2
    
    ax2.fill_between(x * 1e3, 0, energy_density * 1e12, alpha=0.5, color='purple')
    ax2.set_xlabel('Position (mm)', fontsize=11)
    ax2.set_ylabel('Energy Density (pJ/m³)', fontsize=11)
    ax2.set_title('Time-Averaged Energy Distribution', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Mark antinodes (energy maxima)
    for i in range(int(L / (wave["wavelength"] / 2)) + 1):
        antinode_x = i * wave["wavelength"] / 2
        if antinode_x <= L and i % 2 == 0:  # Even multiples for cos pattern
            ax2.axvline(x=antinode_x * 1e3, color='green', alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    
    # Create description
    description_html = f"""
    <h3>Standing Wave Formation</h3>
    <p>Standing waves result from the superposition of two counter-propagating waves with equal amplitude and frequency. They are characterized by stationary nodes and antinodes.</p>
    
    <div class="equation">
        E(x,t) = 2E₀ cos(kx) cos(ωt)
    </div>
    
    <p><strong>Key Features:</strong></p>
    <ul>
        <li>Nodes: Points of zero amplitude (marked in red)</li>
        <li>Antinodes: Points of maximum amplitude</li>
        <li>Node spacing: λ/2 = {wave["wavelength"]/2 * 1e3:.1f} mm</li>
        <li>Energy oscillates between E and B fields but doesn't propagate</li>
    </ul>
    
    <p>Applications include resonant cavities, musical instruments, and laser resonators.</p>
    """
    
    return {
        'image': image_base64,
        'description_html': description_html
    }

def create_interference_visualization(config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Create wave interference visualization"""
    wave = config["physics"]["wave"]
    n_sources = data.get('sources', 2)
    separation = data.get('separation', 2.0) * wave["wavelength"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'Interference Pattern from {n_sources} Coherent Sources', fontsize=14, fontweight='bold')
    
    # Create spatial grid
    extent = 10 * wave["wavelength"]
    resolution = 300
    x = np.linspace(-extent/2, extent/2, resolution)
    y = np.linspace(-extent/2, extent/2, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Calculate source positions
    source_positions = []
    for i in range(n_sources):
        y_pos = (i - (n_sources - 1) / 2) * separation
        source_positions.append((0, y_pos))
    
    # Calculate interference pattern
    E_total = np.zeros_like(X, dtype=complex)
    
    k = wave["wave_number"]
    for x_s, y_s in source_positions:
        R = np.sqrt((X - x_s)**2 + (Y - y_s)**2)
        # Spherical wave with 1/r amplitude decay
        E_total += wave["amplitude"] * np.exp(1j * k * R) / np.sqrt(R + wave["wavelength"]/10)
    
    # Intensity
    intensity = np.abs(E_total)**2
    
    # Normalize for display
    intensity_norm = intensity / np.max(intensity)
    
    # Plot 2D pattern
    im = ax1.imshow(intensity_norm, extent=[-extent/2/wave["wavelength"], extent/2/wave["wavelength"],
                                            -extent/2/wave["wavelength"], extent/2/wave["wavelength"]],
                    cmap='hot', origin='lower', aspect='equal')
    
    # Mark sources
    for x_s, y_s in source_positions:
        ax1.plot(x_s/wave["wavelength"], y_s/wave["wavelength"], 'wo', 
                markersize=10, markeredgecolor='blue', markeredgewidth=2)
    
    ax1.set_xlabel('x (wavelengths)', fontsize=11)
    ax1.set_ylabel('y (wavelengths)', fontsize=11)
    ax1.set_title('Intensity Pattern', fontsize=12)
    ax1.grid(True, alpha=0.3, color='white')
    
    # Plot cross-section
    center_idx = resolution // 2
    intensity_slice = intensity_norm[center_idx, :]
    
    ax2.plot(x/wave["wavelength"], intensity_slice, 'b-', linewidth=2)
    ax2.fill_between(x/wave["wavelength"], 0, intensity_slice, alpha=0.3)
    ax2.set_xlabel('x (wavelengths)', fontsize=11)
    ax2.set_ylabel('Normalized Intensity', fontsize=11)
    ax2.set_title('Intensity Cross-section at y=0', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    
    # For two sources, calculate and show fringe spacing
    if n_sources == 2:
        # Theoretical fringe spacing in far field
        fringe_spacing = wave["wavelength"] / (separation / wave["wavelength"])
        
        # Find actual maxima in the cross-section
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(intensity_slice, height=0.5)
        if len(peaks) > 1:
            actual_spacing = np.mean(np.diff(x[peaks])) / wave["wavelength"]
            ax2.axvline(x=actual_spacing, color='red', linestyle='--', alpha=0.7)
            ax2.text(actual_spacing, 0.9, f'Fringe spacing\n≈ {actual_spacing:.2f}λ', 
                    transform=ax2.transData, ha='left', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    
    # Create description
    description_html = f"""
    <h3>Wave Interference from Coherent Sources</h3>
    <p>Interference occurs when waves from multiple coherent sources superpose. The resulting pattern shows regions of constructive and destructive interference.</p>
    
    <div class="equation">
        I(r) = |E₁(r) + E₂(r) + ... + Eₙ(r)|²
    </div>
    
    <p><strong>Pattern Characteristics:</strong></p>
    <ul>
        <li>Number of sources: {n_sources}</li>
        <li>Source separation: {separation/wave["wavelength"]:.1f} wavelengths</li>
        <li>Bright fringes: constructive interference (waves in phase)</li>
        <li>Dark fringes: destructive interference (waves out of phase)</li>
    </ul>
    
    <p>Applications include Young's double-slit experiment, phased array antennas, and optical interferometry.</p>
    """
    
    return {
        'image': image_base64,
        'description_html': description_html
    }

def create_doppler_visualization(config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Create Doppler effect visualization with correct relativistic formula"""
    wave = config["physics"]["wave"]
    v = data.get('velocity', 0)
    
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.3)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    
    fig.suptitle('Relativistic Doppler Effect for Electromagnetic Waves', fontsize=14, fontweight='bold')
    
    # Check if velocity is valid
    if abs(v) >= SPEED_OF_LIGHT:
        ax1.text(0.5, 0.5, 'Error: Velocity cannot equal or exceed speed of light!', 
                transform=ax1.transAxes, ha='center', va='center', 
                fontsize=16, color='red', weight='bold')
        ax2.axis('off')
        ax3.axis('off')
    else:
        beta = v / SPEED_OF_LIGHT
        
        # Calculate Doppler-shifted frequency
        if v > 0:  # Approaching
            f_observed = wave['frequency'] * np.sqrt((1 + beta) / (1 - beta))
            shift_type = "Blue shift"
            shift_color = 'blue'
        else:  # Receding
            f_observed = wave['frequency'] * np.sqrt((1 - abs(beta)) / (1 + abs(beta)))
            shift_type = "Red shift"
            shift_color = 'red'
        
        lambda_observed = SPEED_OF_LIGHT / f_observed
        
        # Plot wave comparison
        # Use appropriate scale based on wavelength
        if wave["wavelength"] < 1e-6:  # Optical range
            x_max = 20 * wave["wavelength"]
            x_unit = 1e9  # nanometers
            x_label = "Position (nm)"
        elif wave["wavelength"] < 1e-3:  # Infrared to microwave
            x_max = 10 * wave["wavelength"]
            x_unit = 1e6  # micrometers
            x_label = "Position (μm)"
        else:  # Radio waves
            x_max = 5 * wave["wavelength"]
            x_unit = 1e3  # millimeters
            x_label = "Position (mm)"
        
        x = np.linspace(0, x_max, 1000)
        
        # Original wave
        E_original = wave["amplitude"] * np.sin(2 * np.pi * x / wave["wavelength"])
        
        # Doppler-shifted wave
        E_shifted = wave["amplitude"] * np.sin(2 * np.pi * x / lambda_observed)
        
        ax1.plot(x * x_unit, E_original, 'k-', linewidth=2, label='Source wave', alpha=0.6)
        ax1.plot(x * x_unit, E_shifted, color=shift_color, linewidth=2.5, 
                label=f'Observed wave ({shift_type})')
        
        ax1.set_xlabel(x_label, fontsize=11)
        ax1.set_ylabel('Electric Field (V/m)', fontsize=11)
        ax1.set_title('Wave Comparison', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(-1.5 * wave["amplitude"], 1.5 * wave["amplitude"])
        
        # Frequency comparison bar chart
        frequencies = [wave['frequency']/1e9, f_observed/1e9]
        labels = ['Source', 'Observed']
        colors = ['gray', shift_color]
        
        bars = ax2.bar(labels, frequencies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax2.set_ylabel('Frequency (GHz)', fontsize=11)
        ax2.set_title('Frequency Comparison', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar, freq in zip(bars, frequencies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                    f'{freq:.4f}', ha='center', va='bottom', fontsize=10)
        
        # Information panel
        ax3.axis('off')
        info_text = f"""Doppler Analysis
        
Source velocity: {v/1000:.1f} km/s
β = v/c = {beta:.6f}
γ = 1/√(1-β²) = {1/np.sqrt(1-beta**2):.6f}

Source frequency: {wave['frequency']/1e9:.4f} GHz
Source wavelength: {wave['wavelength']*1e3:.3f} mm

Observed frequency: {f_observed/1e9:.4f} GHz
Observed wavelength: {lambda_observed*1e3:.3f} mm

Frequency shift: {(f_observed - wave['frequency'])/1e6:.2f} MHz
Relative shift: {((f_observed/wave['frequency']) - 1)*100:.3f}%

Type: {shift_type}"""
        
        ax3.text(0.05, 0.95, info_text, transform=ax3.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    
    # Create description
    description_html = f"""
    <h3>Relativistic Doppler Effect</h3>
    <p>For electromagnetic waves, the Doppler effect requires relativistic treatment since light always travels at speed c in any reference frame.</p>
    
    <div class="equation">
        f' = f √[(1 + β)/(1 - β)] for approach<br>
        f' = f √[(1 - β)/(1 + β)] for recession<br>
        where β = v/c
    </div>
    
    <p><strong>Physical Significance:</strong></p>
    <ul>
        <li>Blue shift: frequency increases when source approaches</li>
        <li>Red shift: frequency decreases when source recedes</li>
        <li>Used in astronomy to measure stellar velocities</li>
        <li>Basis for radar speed detection and laser velocimetry</li>
    </ul>
    """
    
    return {
        'image': image_base64,
        'description_html': description_html
    }

def create_polarization_visualization(config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Create polarization visualization"""
    wave = config["physics"]["wave"]
    pol_type = data.get('pol_type', 'linear')
    pol_angle = np.radians(data.get('pol_angle', 0))
    
    fig = plt.figure(figsize=(12, 10))
    
    if pol_type == 'linear':
        # Linear polarization visualization
        ax = fig.add_subplot(111, projection='3d')
        
        # Wave propagating in z direction
        z = np.linspace(0, 2 * wave["wavelength"], 200)
        
        # E-field components for linear polarization
        Ex = wave["amplitude"] * np.cos(pol_angle) * np.sin(wave["wave_number"] * z)
        Ey = wave["amplitude"] * np.sin(pol_angle) * np.sin(wave["wave_number"] * z)
        Ez = np.zeros_like(z)
        
        # Plot E-field vector path
        ax.plot(Ex, Ey, z * 1e3, 'b-', linewidth=3, label='E-field trajectory')
        
        # Plot field vectors at several points
        skip = 20
        for i in range(0, len(z), skip):
            if abs(Ex[i]) > 0.01 or abs(Ey[i]) > 0.01:
                ax.quiver(0, 0, z[i]*1e3, Ex[i], Ey[i], 0,
                         color='blue', alpha=0.6, arrow_length_ratio=0.1)
        
        # Plot projections
        ax.plot(Ex, Ey, 0, 'b--', linewidth=2, alpha=0.5)
        ax.plot(Ex, np.zeros_like(Ex), z*1e3, 'r--', alpha=0.5, label='x-projection')
        ax.plot(np.zeros_like(Ey), Ey, z*1e3, 'g--', alpha=0.5, label='y-projection')
        
        # Draw polarization plane
        z_plane = wave["wavelength"] * 1e3
        x_line = np.array([-wave["amplitude"], wave["amplitude"]]) * np.cos(pol_angle)
        y_line = np.array([-wave["amplitude"], wave["amplitude"]]) * np.sin(pol_angle)
        ax.plot(x_line, y_line, [z_plane, z_plane], 'k-', linewidth=3, alpha=0.5)
        
        ax.set_xlabel('Ex (V/m)', fontsize=11)
        ax.set_ylabel('Ey (V/m)', fontsize=11)
        ax.set_zlabel('z (mm)', fontsize=11)
        ax.set_title(f'Linear Polarization at {np.degrees(pol_angle):.0f}°', fontsize=14)
        ax.legend()
        
        # Set equal aspect ratio for x and y
        max_range = wave["amplitude"]
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        
    elif pol_type == 'circular':
        # Circular polarization - create 2x2 subplot
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :], projection='3d')
        
        # Time evolution at fixed position (z=0)
        t = np.linspace(0, 2 * wave["period"], 200)
        omega = wave["angular_frequency"]
        
        Ex_t = wave["amplitude"] * np.cos(omega * t)
        Ey_t = wave["amplitude"] * np.sin(omega * t)
        
        ax1.plot(t * 1e9, Ex_t, 'b-', linewidth=2, label='Ex')
        ax1.plot(t * 1e9, Ey_t, 'r-', linewidth=2, label='Ey')
        ax1.set_xlabel('Time (ns)', fontsize=11)
        ax1.set_ylabel('Field (V/m)', fontsize=11)
        ax1.set_title('E-field Components vs Time at z=0', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(-1.5 * wave["amplitude"], 1.5 * wave["amplitude"])
        
        # Polarization ellipse (circle for circular pol)
        theta = np.linspace(0, 2*np.pi, 100)
        Ex_circle = wave["amplitude"] * np.cos(theta)
        Ey_circle = wave["amplitude"] * np.sin(theta)
        
        ax2.plot(Ex_circle, Ey_circle, 'b-', linewidth=3)
        ax2.arrow(0, 0, Ex_circle[0], Ey_circle[0], 
                 head_width=0.1*wave["amplitude"], head_length=0.08*wave["amplitude"],
                 fc='red', ec='red', linewidth=2)
        ax2.set_xlabel('Ex (V/m)', fontsize=11)
        ax2.set_ylabel('Ey (V/m)', fontsize=11)
        ax2.set_title('E-field Hodograph', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        ax2.set_xlim(-1.5 * wave["amplitude"], 1.5 * wave["amplitude"])
        ax2.set_ylim(-1.5 * wave["amplitude"], 1.5 * wave["amplitude"])
        
        # 3D helix visualization
        z = np.linspace(0, 2 * wave["wavelength"], 400)
        k = wave["wave_number"]
        
        Ex_3d = wave["amplitude"] * np.cos(k * z)
        Ey_3d = wave["amplitude"] * np.sin(k * z)
        
        ax3.plot(Ex_3d, Ey_3d, z * 1e3, 'b-', linewidth=3)
        
        # Add some field vectors
        skip = 40
        for i in range(0, len(z), skip):
            ax3.quiver(0, 0, z[i]*1e3, Ex_3d[i], Ey_3d[i], 0,
                      color='blue', alpha=0.6, arrow_length_ratio=0.1)
        
        ax3.set_xlabel('Ex (V/m)', fontsize=11)
        ax3.set_ylabel('Ey (V/m)', fontsize=11)
        ax3.set_zlabel('z (mm)', fontsize=11)
        ax3.set_title('Circular Polarization - Helical E-field', fontsize=12)
        
        max_range = wave["amplitude"]
        ax3.set_xlim([-max_range, max_range])
        ax3.set_ylim([-max_range, max_range])
    
    elif pol_type == 'elliptical':
        # Elliptical polarization
        ax = fig.add_subplot(111, projection='3d')
        
        # Parameters for elliptical polarization
        a = wave["amplitude"]  # Major axis
        b = wave["amplitude"] * 0.5  # Minor axis
        
        z = np.linspace(0, 2 * wave["wavelength"], 400)
        k = wave["wave_number"]
        
        # Elliptical field components
        Ex = a * np.cos(pol_angle) * np.cos(k * z) - b * np.sin(pol_angle) * np.sin(k * z)
        Ey = a * np.sin(pol_angle) * np.cos(k * z) + b * np.cos(pol_angle) * np.sin(k * z)
        
        ax.plot(Ex, Ey, z * 1e3, 'b-', linewidth=3, label='E-field trajectory')
        
        # Plot projections
        ax.plot(Ex, Ey, 0, 'b--', linewidth=2, alpha=0.5)
        
        # Add field vectors
        skip = 40
        for i in range(0, len(z), skip):
            if abs(Ex[i]) > 0.01 or abs(Ey[i]) > 0.01:
                ax.quiver(0, 0, z[i]*1e3, Ex[i], Ey[i], 0,
                         color='blue', alpha=0.6, arrow_length_ratio=0.1)
        
        ax.set_xlabel('Ex (V/m)', fontsize=11)
        ax.set_ylabel('Ey (V/m)', fontsize=11)
        ax.set_zlabel('z (mm)', fontsize=11)
        ax.set_title(f'Elliptical Polarization at {np.degrees(pol_angle):.0f}°', fontsize=14)
        ax.legend()
        
        max_range = wave["amplitude"]
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    
    # Create description
    description_html = f"""
    <h3>Electromagnetic Wave Polarization</h3>
    <p>Polarization describes the orientation and behavior of the electric field vector as the wave propagates.</p>
    
    <div class="equation">
        Linear: E = E₀(x̂ cos α + ŷ sin α) cos(kz - ωt)<br>
        Circular: E = E₀(x̂ cos(kz - ωt) ± ŷ sin(kz - ωt))
    </div>
    
    <p><strong>Polarization Types:</strong></p>
    <ul>
        <li><strong>Linear:</strong> E-field oscillates in a fixed plane</li>
        <li><strong>Circular:</strong> E-field rotates with constant magnitude</li>
        <li><strong>Elliptical:</strong> General case between linear and circular</li>
    </ul>
    
    <p>Applications include LCD displays, 3D glasses, satellite communications, and optical isolators.</p>
    """
    
    return {
        'image': image_base64,
        'description_html': description_html
    }

def create_dipole_visualization(config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Create dipole antenna radiation pattern visualization"""
    wave = config["physics"]["wave"]
    dipole_length = data.get('dipole_length', 0.5) * wave["wavelength"]
    current_dist = data.get('current_dist', 'uniform')
    view_type = data.get('view_type', '3d')
    
    if view_type == '3d':
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create spherical coordinate grid
        theta = np.linspace(0, np.pi, 100)
        phi = np.linspace(0, 2*np.pi, 100)
        THETA, PHI = np.meshgrid(theta, phi)
        
        # Calculate radiation pattern
        k = wave["wave_number"]
        l = dipole_length
        
        # Different patterns for different lengths
        if abs(l / wave["wavelength"] - 0.5) < 0.1:
            # Half-wave dipole
            F = np.abs(np.cos((np.pi/2) * np.cos(THETA)) / np.sin(THETA + 1e-10))
        elif l / wave["wavelength"] < 0.1:
            # Short dipole
            F = np.abs(np.sin(THETA))
        else:
            # General dipole
            kl = k * l
            F = np.abs((np.cos(kl/2 * np.cos(THETA)) - np.cos(kl/2)) / (np.sin(THETA) + 1e-10))
        
        # Normalize
        F = F / np.max(F)
        
        # Convert to Cartesian coordinates
        R = F
        X = R * np.sin(THETA) * np.cos(PHI)
        Y = R * np.sin(THETA) * np.sin(PHI)
        Z = R * np.cos(THETA)
        
        # Plot 3D surface
        surf = ax.plot_surface(X, Y, Z, cmap='jet', alpha=0.8, 
                              linewidth=0, antialiased=True)
        
        # Add dipole representation
        dipole_z = np.linspace(-l/wave["wavelength"]/2, l/wave["wavelength"]/2, 20)
        ax.plot([0]*len(dipole_z), [0]*len(dipole_z), dipole_z, 
               'k-', linewidth=4, label='Dipole')
        
        ax.set_xlabel('x/λ', fontsize=11)
        ax.set_ylabel('y/λ', fontsize=11)
        ax.set_zlabel('z/λ', fontsize=11)
        ax.set_title(f'3D Radiation Pattern - {dipole_length/wave["wavelength"]:.1f}λ Dipole', 
                    fontsize=14)
        
        # Set equal aspect ratio
        max_range = 1.2
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
    else:
        # 2D pattern (E-plane or H-plane)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        if view_type == '2d_elevation':
            # E-plane (elevation pattern)
            theta = np.linspace(0, 2*np.pi, 1000)
            
            # Calculate pattern
            k = wave["wave_number"]
            l = dipole_length
            
            # Adjust theta for proper orientation
            theta_calc = theta.copy()
            theta_calc[theta > np.pi] = 2*np.pi - theta[theta > np.pi]
            
            if abs(l / wave["wavelength"] - 0.5) < 0.1:
                # Half-wave dipole
                F = np.abs(np.cos((np.pi/2) * np.cos(theta_calc)) / (np.sin(theta_calc) + 1e-10))
            elif l / wave["wavelength"] < 0.1:
                # Short dipole
                F = np.abs(np.sin(theta_calc))
            else:
                # General dipole
                kl = k * l
                F = np.abs((np.cos(kl/2 * np.cos(theta_calc)) - np.cos(kl/2)) / 
                          (np.sin(theta_calc) + 1e-10))
            
            # Normalize
            F = F / (np.max(F) + 1e-10)
            
            # Polar plot
            ax1.plot(theta, F, 'b-', linewidth=2)
            ax1.fill(theta, F, alpha=0.3)
            ax1.set_theta_zero_location('N')
            ax1.set_theta_direction(-1)
            ax1.set_title('E-plane (Elevation) Pattern', fontsize=12)
            ax1.grid(True)
            
            # Cartesian plot
            theta_deg = np.degrees(theta)
            ax2.plot(theta_deg, F, 'b-', linewidth=2)
            ax2.fill_between(theta_deg, 0, F, alpha=0.3)
            ax2.set_xlabel('Angle (degrees)', fontsize=11)
            ax2.set_ylabel('Normalized Pattern', fontsize=11)
            ax2.set_title('E-plane Pattern (Linear Scale)', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim([0, 360])
            ax2.set_ylim([0, 1.1])
            
        else:  # H-plane
            # H-plane pattern (omnidirectional for vertical dipole)
            phi = np.linspace(0, 2*np.pi, 1000)
            F = np.ones_like(phi)  # Omnidirectional in H-plane
            
            # Polar plot
            ax1.plot(phi, F, 'r-', linewidth=2)
            ax1.fill(phi, F, alpha=0.3, color='red')
            ax1.set_theta_zero_location('N')
            ax1.set_theta_direction(-1)
            ax1.set_title('H-plane (Azimuth) Pattern', fontsize=12)
            ax1.grid(True)
            ax1.set_ylim([0, 1.2])
            
            # Info panel
            ax2.axis('off')
            info_text = f"""Dipole Antenna Analysis
            
Length: {dipole_length/wave["wavelength"]:.2f}λ = {dipole_length*1e3:.1f} mm
Frequency: {wave['frequency']/1e9:.3f} GHz
Current distribution: {current_dist}

Radiation resistance: {config['physics']['dipole']['radiation_resistance']:.1f} Ω

Pattern characteristics:
- E-plane: Figure-8 pattern
- H-plane: Omnidirectional
- Maximum radiation: θ = 90° (broadside)
- Null radiation: θ = 0°, 180° (endfire)

Directivity (approx):
- Short dipole: 1.5 (1.76 dBi)
- Half-wave dipole: 1.64 (2.15 dBi)
- Full-wave dipole: 2.41 (3.82 dBi)"""
            
            ax2.text(0.1, 0.9, info_text, transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    
    # Create description
    description_html = f"""
    <h3>Dipole Antenna Radiation Pattern</h3>
    <p>The dipole antenna is a fundamental radiating element consisting of a linear conductor fed at the center. The radiation pattern depends on the dipole length relative to wavelength.</p>
    
    <div class="equation">
        E<sub>θ</sub> = (jηI₀/2πr) × f(θ) × e<sup>-jkr</sup><br>
        where f(θ) is the pattern function
    </div>
    
    <p><strong>Key Characteristics:</strong></p>
    <ul>
        <li>Length: {dipole_length/wave["wavelength"]:.2f}λ</li>
        <li>Radiation resistance: {config['physics']['dipole']['radiation_resistance']:.1f} Ω</li>
        <li>E-plane pattern: Figure-8 shape with nulls along axis</li>
        <li>H-plane pattern: Omnidirectional (circular)</li>
        <li>Polarization: Linear, aligned with dipole axis</li>
    </ul>
    
    <p>Applications include FM radio antennas, Wi-Fi antennas, and as feed elements in more complex antenna systems.</p>
    """
    
    return {
        'image': image_base64,
        'description_html': description_html
    }

def create_waveguide_visualization(config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Create waveguide mode propagation visualization"""
    wave = config["physics"]["wave"]
    waveguide = config["physics"]["waveguide"]
    
    if waveguide.get("evanescent", False):
        # Evanescent mode visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        z = np.linspace(0, 5 * wave["wavelength"], 1000)
        alpha = waveguide["attenuation_constant"]
        
        # Exponentially decaying field
        E = wave["amplitude"] * np.exp(-alpha * z)
        
        ax.plot(z * 1e3, E, 'b-', linewidth=2.5, label='Field amplitude')
        ax.fill_between(z * 1e3, 0, E, alpha=0.3)
        ax.axhline(y=wave["amplitude"] * np.exp(-1), color='r', linestyle='--', 
                  label=f'1/e point: {1/alpha*1e3:.1f} mm')
        
        ax.set_xlabel('Position z (mm)', fontsize=11)
        ax.set_ylabel('Field Amplitude (V/m)', fontsize=11)
        ax.set_title(f'Evanescent Mode - Below Cutoff Frequency', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add text box with information
        info_text = f"""Evanescent Mode
        
Operating frequency: {wave['frequency']/1e9:.3f} GHz
Cutoff frequency: {waveguide['cutoff_frequency']/1e9:.3f} GHz
f/fc = {wave['frequency']/waveguide['cutoff_frequency']:.3f} < 1

The mode cannot propagate and
decays exponentially with distance.

Attenuation: {alpha:.1f} Np/m
Decay length: {1/alpha*1e3:.1f} mm"""
        
        ax.text(0.98, 0.95, info_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
    else:
        # Propagating mode visualization
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        
        a = waveguide["width"]
        b = waveguide["height"]
        mode = data.get("mode", "TE10")
        m = waveguide["mode_indices"]["m"]
        n = waveguide["mode_indices"]["n"]
        
        # Create 2D grid for field visualization
        x = np.linspace(0, a, 100)
        y = np.linspace(0, b, 100)
        X, Y = np.meshgrid(x, y)
        
        if waveguide["mode_type"] == "TE":
            # TE modes - Ez = 0
            if mode == "TE10":
                # Dominant mode
                Ey = wave["amplitude"] * np.sin(np.pi * X / a)
                Ex = np.zeros_like(Ey)
                Ez = np.zeros_like(Ey)
            elif mode == "TE20":
                Ey = wave["amplitude"] * np.sin(2 * np.pi * X / a)
                Ex = np.zeros_like(Ey)
                Ez = np.zeros_like(Ey)
            elif mode == "TE01":
                Ex = wave["amplitude"] * np.sin(np.pi * Y / b)
                Ey = np.zeros_like(Ex)
                Ez = np.zeros_like(Ex)
            elif mode == "TE11":
                # TE11 has both Ex and Ey components
                Ex = wave["amplitude"] * np.cos(np.pi * X / a) * np.sin(np.pi * Y / b)
                Ey = -wave["amplitude"] * np.sin(np.pi * X / a) * np.cos(np.pi * Y / b)
                Ez = np.zeros_like(Ex)
            else:
                # General TE_mn mode
                Ey = wave["amplitude"] * np.sin(m * np.pi * X / a) * np.cos(n * np.pi * Y / b)
                Ex = -wave["amplitude"] * np.cos(m * np.pi * X / a) * np.sin(n * np.pi * Y / b)
                Ez = np.zeros_like(Ex)
        else:
            # TM modes - Hz = 0
            if mode == "TM11":
                Ez = wave["amplitude"] * np.sin(np.pi * X / a) * np.sin(np.pi * Y / b)
                kc = np.sqrt((np.pi/a)**2 + (np.pi/b)**2)
                Ex = -(np.pi/a) / kc**2 * wave["amplitude"] * np.cos(np.pi * X / a) * np.sin(np.pi * Y / b)
                Ey = -(np.pi/b) / kc**2 * wave["amplitude"] * np.sin(np.pi * X / a) * np.cos(np.pi * Y / b)
        
        # Calculate field magnitude
        E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)
        
        # Plot field distribution
        im = ax1.contourf(X * 1e3, Y * 1e3, E_mag, levels=50, cmap='jet')
        ax1.set_xlabel('x (mm)', fontsize=11)
        ax1.set_ylabel('y (mm)', fontsize=11)
        ax1.set_title(f'{mode} Mode Field Distribution at z=0', fontsize=14)
        ax1.set_aspect('equal')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1, orientation='vertical', pad=0.02)
        cbar.set_label('|E| (V/m)', fontsize=11)
        
        # Add waveguide walls
        rect = plt.Rectangle((0, 0), a*1e3, b*1e3, fill=False, edgecolor='black', linewidth=2)
        ax1.add_patch(rect)
        
        # Add field vectors (skip some points for clarity)
        skip = 10
        X_vec = X[::skip, ::skip]
        Y_vec = Y[::skip, ::skip]
        Ex_vec = Ex[::skip, ::skip]
        Ey_vec = Ey[::skip, ::skip]
        
        # Normalize vectors for display
        E_vec_mag = np.sqrt(Ex_vec**2 + Ey_vec**2)
        mask = E_vec_mag > 0.1 * np.max(E_vec_mag)
        
        if np.any(mask):
            ax1.quiver(X_vec[mask]*1e3, Y_vec[mask]*1e3, 
                      Ex_vec[mask]/E_vec_mag[mask], Ey_vec[mask]/E_vec_mag[mask],
                      E_vec_mag[mask], cmap='jet', scale=30, width=0.003, alpha=0.7)
        
        # Dispersion diagram
        fc = waveguide["cutoff_frequency"]
        f_range = np.linspace(fc * 0.5, fc * 2.5, 1000)
        
        # Calculate propagation constant vs frequency
        beta_range = np.zeros_like(f_range)
        for i, f in enumerate(f_range):
            if f > fc:
                beta_range[i] = 2 * np.pi * f / SPEED_OF_LIGHT * np.sqrt(1 - (fc/f)**2)
        
        ax2.plot(f_range/1e9, beta_range, 'b-', linewidth=2)
        ax2.axvline(x=fc/1e9, color='r', linestyle='--', label='Cutoff frequency')
        ax2.axvline(x=wave['frequency']/1e9, color='g', linestyle='--', label='Operating frequency')
        ax2.set_xlabel('Frequency (GHz)', fontsize=11)
        ax2.set_ylabel('β (rad/m)', fontsize=11)
        ax2.set_title('Dispersion Relation', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Information panel
        ax3.axis('off')
        info_text = f"""Waveguide Mode Analysis
        
Mode: {mode}
Type: {waveguide['mode_type']}
Dimensions: {a*1e3:.1f} × {b*1e3:.1f} mm

Operating frequency: {wave['frequency']/1e9:.3f} GHz
Cutoff frequency: {fc/1e9:.3f} GHz
f/fc = {wave['frequency']/fc:.3f}

Propagation constant: {waveguide['propagation_constant']:.1f} rad/m
Guide wavelength: {waveguide['guide_wavelength']*1e3:.1f} mm
Phase velocity: {waveguide['phase_velocity']/SPEED_OF_LIGHT:.3f}c
Group velocity: {waveguide['group_velocity']/SPEED_OF_LIGHT:.3f}c
Wave impedance: {waveguide['impedance']:.1f} Ω"""
        
        ax3.text(0.1, 0.9, info_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    fig.suptitle(f'Rectangular Waveguide - {data.get("mode", "TE10")} Mode', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    
    # Create description
    description_html = f"""
    <h3>Waveguide Mode Propagation</h3>
    <p>Waveguides confine and guide electromagnetic waves through total internal reflection. Each mode has a specific field distribution and cutoff frequency.</p>
    
    <div class="equation">
        Cutoff frequency: f<sub>c</sub> = (c/2π)√[(mπ/a)² + (nπ/b)²]<br>
        Propagation constant: β = (2πf/c)√[1 - (f<sub>c</sub>/f)²]
    </div>
    
    <p><strong>Mode Characteristics:</strong></p>
    <ul>
        <li>TE modes: Transverse Electric (E<sub>z</sub> = 0)</li>
        <li>TM modes: Transverse Magnetic (H<sub>z</sub> = 0)</li>
        <li>Dominant mode: TE<sub>10</sub> has lowest cutoff frequency</li>
        <li>Single-mode operation: f<sub>c,TE10</sub> < f < f<sub>c,TE20</sub></li>
    </ul>
    
    <p>Applications include microwave transmission lines, radar systems, and satellite communications.</p>
    """
    
    return {
        'image': image_base64,
        'description_html': description_html
    }

if __name__ == '__main__':
    print("\nProfessional EM Wave Visualizer - Complete University Edition")
    print("Features:")
    print("- Plane wave propagation")
    print("- Reflection and refraction with Fresnel coefficients")
    print("- Standing wave formation")
    print("- Multi-source interference patterns")
    print("- Relativistic Doppler effect")
    print("- Polarization states (linear, circular, elliptical)")
    print("- Dipole antenna radiation patterns")
    print("- Waveguide mode propagation")
    print("\nStarting server at http://localhost:5000\n")
    app.run(debug=True, port=5000)