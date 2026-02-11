import librosa
from src.filters import lowpass_fir, highpass_fir, apply_filter
from src.analysis import calculate_fft
from src.visualization import (plot_audio_effects_comparison, plot_waveform, plot_comparison, plot_filters_comparison)

# Configuración
AUDIO_PATH = 'audio_samples/sample-15s.wav'
CUTOFF_FREQ = 1000  # Hz
NUM_TAPS = 101

# Cargar audio
print("Cargando audio...")
y, sr = librosa.load(AUDIO_PATH, sr=None)
print(f"Frecuencia de muestreo: {sr} Hz")
print(f"Duración: {len(y)/sr:.2f} segundos")
print(f"Muestras: {len(y)}")

# Visualizar señal original
plot_waveform(y, sr, title="Señal original")

# Diseñar filtros
print(f"\nDiseñando filtros (fc={CUTOFF_FREQ} Hz)...")
h_low = lowpass_fir(fc=CUTOFF_FREQ, fs=sr, num_taps=NUM_TAPS)
h_high = highpass_fir(fc=CUTOFF_FREQ, fs=sr, num_taps=NUM_TAPS)
print(f"Filtros creados con {len(h_low)} coeficientes")

# Comparación de filtros
plot_filters_comparison(h_low, h_high, fc=CUTOFF_FREQ, sr=sr)

# Aplicar filtros
print("\nAplicando filtros...")
y_lowpass = apply_filter(y, h_low)
y_highpass = apply_filter(y, h_high)

# Análisis espectral
print("\nCalculando espectros...")
freqs_orig, _, mag_orig_db = calculate_fft(y, sr)
_, _, mag_low_db = calculate_fft(y_lowpass, sr)
_, _, mag_high_db = calculate_fft(y_highpass, sr)

# Comparación espectral de filtros
plot_audio_effects_comparison(freqs_orig, mag_orig_db, mag_low_db, 
                              mag_high_db, fc=CUTOFF_FREQ)

print("\n✓ Análisis completado")