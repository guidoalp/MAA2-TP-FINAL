import librosa
from src.filters import lowpass_fir, apply_filter
from src.analysis import calculate_fft, calculate_filter_response
from src.visualization import (plot_waveform, plot_comparison,
                               plot_filter_response, plot_impulse_response)

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

# Diseñar filtro
print(f"\nDiseñando filtro paso bajo (fc={CUTOFF_FREQ} Hz)...")
h = lowpass_fir(fc=CUTOFF_FREQ, fs=sr, num_taps=NUM_TAPS)
print(f"Filtro creado con {len(h)} coeficientes")

# Visualizar respuesta al impulso
plot_impulse_response(h, title=f"Respuesta al impulso (fc={CUTOFF_FREQ} Hz)")

# Aplicar filtro
print("\nAplicando filtro...")
y_filtrado = apply_filter(y, h)

# Análisis espectral
print("\nCalculando espectros...")
freqs_orig, _, mag_orig_db = calculate_fft(y, sr)
freqs_filt, _, mag_filt_db = calculate_fft(y_filtrado, sr)

# Visualizaciones comparativas
plot_comparison(freqs_orig, mag_orig_db, mag_filt_db, fc=CUTOFF_FREQ)

# Respuesta en frecuencia del filtro
freqs_filter, mag_filter_db = calculate_filter_response(h, sr)
plot_filter_response(freqs_filter, mag_filter_db, fc=CUTOFF_FREQ)

print("\n✓ Análisis completado")