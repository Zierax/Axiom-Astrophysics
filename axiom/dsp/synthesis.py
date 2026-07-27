import random

import numpy as np


def generate_noise(length, amplitude=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.normal(0, amplitude, length)

def synthesize_pulsar(length=256, period_sec=2.0, pulse_width_sec=0.1, intensity=5.0, sample_rate_hz=1000.0, seed=None):
    """
    Synthesize periodic Gaussian pulses representing a pulsar.
    """
    if seed is not None:
        np.random.seed(seed)
        
    t = np.arange(length) / sample_rate_hz
    waveform = np.zeros(length)
    
    # Calculate phase
    phase = (t % period_sec) / period_sec
    pulse_phase_width = pulse_width_sec / period_sec
    
    # Center the pulse at phase 0.5 for convenience
    pulse_center = 0.5
    dist = np.minimum(np.abs(phase - pulse_center), 1.0 - np.abs(phase - pulse_center))
    
    # Gaussian pulse profile
    sigma = pulse_phase_width / 2.0
    if sigma > 0:
        waveform = intensity * np.exp(-0.5 * (dist / sigma)**2)
        
    # Add background noise
    waveform += np.random.normal(0, 0.2, length)
    return waveform

def synthesize_frb(length=256, width_ms=5.0, dm=300.0, intensity=25.0, sample_rate_hz=1000.0, seed=None):
    """
    Synthesize a single fast radio burst pulse with dispersion broadening and scattering tail.
    """
    if seed is not None:
        np.random.seed(seed)
        
    t = np.arange(length) / sample_rate_hz
    waveform = np.zeros(length)
    
    # Center the pulse in the time series
    pulse_center_t = 0.128  # seconds (middle of 256ms at 1kHz)
    
    # Dispersion broadening is proportional to DM. Width is widened.
    width_sec = width_ms / 1000.0
    dispersed_width = width_sec * (1.0 + 0.005 * dm)
    
    # Gaussian pulse + exponential scattering tail
    sigma = dispersed_width / 2.0
    dist = t - pulse_center_t
    
    # Before peak: Gaussian. After peak: Gaussian convoluted with exponential tail.
    for i, dt in enumerate(dist):
        if dt < 0:
            if sigma > 0:
                waveform[i] = intensity * np.exp(-0.5 * (dt / sigma)**2)
        else:
            # Scattering tail lifetime increases with DM
            tau = 0.01 * (1.0 + 0.008 * dm)
            waveform[i] = intensity * np.exp(-0.5 * (dt / sigma)**2) * np.exp(-dt / tau)
            
    waveform += np.random.normal(0, 0.2, length)
    return waveform

def synthesize_hi_line(length=256, freq_mhz=1420.405, intensity=3.0, sample_rate_hz=1000.0, seed=None):
    """
    Synthesize a neutral hydrogen (HI) spectral line profile in frequency domain.
    """
    if seed is not None:
        np.random.seed(seed)
        
    # HI line: Gaussian profile centered in the frequency band
    # Simulate a narrowband frequency profile
    f_center = freq_mhz % 10.0  # Normalize to fit inside simulated relative band
    f = np.linspace(f_center - 2.0, f_center + 2.0, length)
    
    # Gaussian emission peak
    sigma = 0.15  # Line width
    waveform = intensity * np.exp(-0.5 * ((f - f_center) / sigma)**2)
    
    waveform += np.random.normal(0, 0.1, length)
    return waveform

def synthesize_quasar(length=256, intensity=10.0, seed=None):
    """
    Synthesize a stochastic red noise time series (random walk) representing quasar flux variability.
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Generate random walk: cumulative sum of random steps
    steps = np.random.normal(0, 1.0, length)
    waveform = np.cumsum(steps)
    
    # Normalize to zero mean and scale to intensity
    waveform = waveform - np.mean(waveform)
    std = np.std(waveform)
    if std > 0:
        waveform = intensity * (waveform / std)
        
    # Add noise
    waveform += np.random.normal(0, 0.2, length)
    return waveform

def synthesize_rfi(length=256, rfi_type="CW", intensity=15.0, sample_rate_hz=1000.0, seed=None):
    """
    Synthesize terrestrial radio frequency interference.
    Types: 'CW' (continuous wave tone), 'Sweep' (drifting tone), 'Burst' (pulsed noise).
    """
    if seed is not None:
        np.random.seed(seed)
        
    t = np.arange(length) / sample_rate_hz
    waveform = np.zeros(length)
    
    if rfi_type == "CW":
        # Pure continuous tone at e.g. 50 Hz
        f = 50.0
        waveform = intensity * np.sin(2.0 * np.pi * f * t)
    elif rfi_type == "Sweep":
        # Drifting frequency tone (chirp)
        f0 = 20.0
        beta = 150.0  # sweep rate
        waveform = intensity * np.sin(2.0 * np.pi * (f0 * t + 0.5 * beta * t**2))
    elif rfi_type == "Burst":
        # Packet-like noise bursts
        waveform = np.random.normal(0, intensity, length)
        # Apply envelope (duty cycle of burst)
        envelope = np.zeros(length)
        # 3 bursts
        for start, end in [(20, 60), (100, 140), (180, 220)]:
            envelope[start:end] = 1.0
        waveform = waveform * envelope
        
    waveform += np.random.normal(0, 0.1, length)
    return waveform

def synthesize_wow(length=256, intensity=30.0, sample_rate_hz=1000.0, seed=None):
    """
    Synthesize the Wow! signal anomaly: Gaussian transit envelope modulated with a continuous tone.
    """
    if seed is not None:
        np.random.seed(seed)
        
    t = np.arange(length) / sample_rate_hz
    
    # Gaussian beam transit over 256 samples
    center_t = 0.128
    sigma_t = 0.045
    transit_envelope = np.exp(-0.5 * ((t - center_t) / sigma_t)**2)
    
    # Modulation tone (narrowband continuous carrier at e.g. 80 Hz)
    carrier = np.sin(2.0 * np.pi * 80.0 * t)
    
    waveform = intensity * transit_envelope * carrier
    waveform += np.random.normal(0, 0.1, length)
    return waveform

def synthesize_blc1(length=256, intensity=15.0, sample_rate_hz=1000.0, seed=None):
    """
    Synthesize BLC1 anomaly: narrow frequency tone with slow linear drift.
    """
    if seed is not None:
        np.random.seed(seed)
        
    t = np.arange(length) / sample_rate_hz
    
    # Slow drift: frequency shifts linearly over time
    f0 = 120.0
    drift_rate = 12.0  # Hz/s (slow, but noticeable over length)
    phase = 2.0 * np.pi * (f0 * t + 0.5 * drift_rate * t**2)
    
    waveform = intensity * np.sin(phase)
    waveform += np.random.normal(0, 0.1, length)
    return waveform

def synthesize_arecibo(length=256, intensity=22.0, sample_rate_hz=1000.0, seed=None):
    """
    Synthesize Arecibo message echo anomaly: binary frequency shift keying (FSK).
    """
    if seed is not None:
        np.random.seed(seed)
        
    t = np.arange(length) / sample_rate_hz
    waveform = np.zeros(length)
    
    # Simple binary message sequence (7 bits)
    bits = [1, 0, 1, 1, 0, 1, 0]
    samples_per_bit = length // len(bits)
    
    for i, bit in enumerate(bits):
        start = i * samples_per_bit
        end = min((i + 1) * samples_per_bit, length)
        # Shift frequency depending on bit state
        freq = 150.0 if bit == 1 else 90.0
        waveform[start:end] = intensity * np.sin(2.0 * np.pi * freq * t[start:end])
        
    waveform += np.random.normal(0, 0.2, length)
    return waveform

def generate_waveform_by_class(origin_class, signal_id, freq_mhz=1420.4, intensity=10.0, seed=None):
    """
    Dispatcher to generate a 256-sample physical waveform vector matching the source class or specific anomaly ID.
    """
    # Check for specific historical anomalies first
    if signal_id == "ANOMALY_WOW_1977":
        return synthesize_wow(intensity=intensity, seed=seed)
    elif signal_id == "ANOMALY_BLC1_2020":
        return synthesize_blc1(intensity=intensity, seed=seed)
    elif signal_id == "ANOMALY_ARECIBO_ECHO":
        return synthesize_arecibo(intensity=intensity, seed=seed)
    elif signal_id.startswith("ANOMALY_LORIMER") or signal_id.startswith("ANOMALY_FRB"):
        return synthesize_frb(dm=500.0, intensity=intensity, seed=seed)
    elif signal_id.startswith("ANOMALY_SHGb02") or signal_id.startswith("ANOMALY_HD"):
        return synthesize_blc1(intensity=intensity, seed=seed)
        
    # Otherwise generate based on class type
    if origin_class == "Pulsar" or "PUL" in signal_id:
        return synthesize_pulsar(intensity=intensity, seed=seed)
    elif origin_class == "FRB" or "FRB" in signal_id:
        return synthesize_frb(intensity=intensity, seed=seed)
    elif origin_class == "Hydrogen" or "HYD" in signal_id or abs(freq_mhz - 1420.4) < 1.0:
        return synthesize_hi_line(freq_mhz=freq_mhz, intensity=intensity, seed=seed)
    elif origin_class == "Quasar" or "QUA" in signal_id or "NED" in signal_id:
        return synthesize_quasar(intensity=intensity, seed=seed)
    elif origin_class == "Interference" or "RFI" in signal_id:
        # Choose a random RFI subtype
        rfi_type = random.choice(["CW", "Sweep", "Burst"])
        return synthesize_rfi(rfi_type=rfi_type, intensity=intensity, seed=seed)
    else:
        # Default fallback is noise-modulated baseline
        return generate_noise(256, amplitude=intensity/5.0, seed=seed)

def synthesize_from_htru2(profile_mean, profile_std, profile_skew, profile_kurt, class_label, seed=None):
    """
    Generates a 256-sample waveform that has the exact physical statistics (mean and std)
    of a real HTRU2 candidate. Uses a Cornish-Fisher expansion to modulate the underlying
    carrier distribution based on target skewness and kurtosis parameters.
    """
    if seed is not None:
        np.random.seed(seed)
        
    length = 256
    
    # 1. Base carrier generation
    if class_label == 1:
        # Pulsar carrier: Gaussian pulse train with random period and noise
        period = np.random.uniform(2.0, 5.0)
        pulse_width = np.random.uniform(0.1, 0.3)
        t = np.linspace(0, 10, length)
        phase = (t % period) / period
        dist = np.minimum(np.abs(phase - 0.5), 1.0 - np.abs(phase - 0.5))
        sigma = (pulse_width / period) / 2.0
        raw_wave = np.exp(-0.5 * (dist / sigma)**2) if sigma > 0 else np.zeros(length)
        # Add baseline noise to allow Cornish-Fisher expansion to modify statistical structure
        raw_wave += np.random.normal(0, 0.1, length)
    else:
        # RFI/Noise carrier: continuous wave carrier with noise
        t = np.linspace(0, 1, length)
        freq = np.random.choice([50.0, 120.0, 240.0])
        raw_wave = np.sin(2.0 * np.pi * freq * t)
        raw_wave += np.random.normal(0, 0.3, length)

    # 2. Standardize base wave
    mean_raw = np.mean(raw_wave)
    std_raw = np.std(raw_wave)
    if std_raw > 0:
        x = (raw_wave - mean_raw) / std_raw
    else:
        x = np.random.normal(0, 1, length)

    # 3. Cornish-Fisher expansion for skewness and kurtosis
    # Skew transform
    y = x + (profile_skew / 6.0) * (x**2 - 1.0)
    
    # Kurtosis transform
    z = y + (profile_kurt / 24.0) * (y**3 - 3.0 * y)
    
    # 4. Final scale and shift to match targeted moments exactly
    mean_z = np.mean(z)
    std_z = np.std(z)
    
    if std_z > 0:
        z_standard = (z - mean_z) / std_z
    else:
        z_standard = np.random.normal(0, 1, length)
        
    w_final = profile_mean + z_standard * profile_std
    return w_final
