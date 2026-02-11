import matplotlib.pyplot as plt
import numpy as np

def plot_waveform(signal, sr, title="Forma de onda"):
    """Grafica la forma de onda de una señal."""
    tiempo = np.linspace(0, len(signal)/sr, len(signal))
    
    plt.figure(figsize=(12, 4))
    plt.plot(tiempo, signal)
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_spectrum(frequencies, magnitude_db, fc=None, f_max=10000, title="Espectro de magnitud"):
    """
    Grafica el espectro de una señal.
    
    Parámetros:
    -----------
    frequencies (ndarray): Array de frecuencias
    magnitude_db (ndarray): Magnitud en dB
    fc (float) (opcional): Frecuencia de corte para la marca
    f_max (float): Frecuencia máxima a mostrar
    """
    idx_max = np.where(frequencies >= f_max)[0][0]
    
    plt.figure(figsize=(12, 4))
    plt.plot(frequencies[1:idx_max], magnitude_db[1:idx_max])
    
    if fc is not None:
        plt.axvline(x=fc, color='red', linestyle='--', 
                   label=f'Frecuencia de corte ({fc} Hz)')
        plt.legend()
    
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud (dB)')
    plt.title(title)
    plt.grid(True)
    plt.xlim(0, f_max)
    plt.show()


def plot_comparison(frequencies, mag_original_db, mag_filtered_db, fc, f_max=5000):
    """
    Grafica comparación espectral y diferencia entre señales.
    
    Parametros:
    -----------
    frequencies (ndarray): Array de frecuencias
    mag_original_db (ndarray): Magnitud original en dB
    mag_filtered_db (ndarray): Magnitud filtrada en dB
    fc (float): Frecuencia de corte
    f_max: (float): Frecuencia máxima a mostrar
    """
    idx_max = np.where(frequencies >= f_max)[0][0]
    
    plt.figure(figsize=(12, 8))
    
    # Espectros superpuestos
    plt.subplot(2, 1, 1)
    plt.plot(frequencies[1:idx_max], mag_original_db[1:idx_max],
             alpha=0.7, label='Original')
    plt.plot(frequencies[1:idx_max], mag_filtered_db[1:idx_max],
             alpha=0.7, label='Filtrado')
    plt.axvline(x=fc, color='red', linestyle='--', 
               label='Frecuencia de corte')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud (dB)')
    plt.title('Comparación Espectral')
    plt.legend()
    plt.grid(True)
    
    # Diferencia espectral
    plt.subplot(2, 1, 2)
    diferencia_db = mag_original_db - mag_filtered_db
    plt.plot(frequencies[1:idx_max], diferencia_db[1:idx_max], 
            color='green')
    plt.axvline(x=fc, color='red', linestyle='--', 
               label='Frecuencia de corte')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Atenuación (dB)')
    plt.title('Diferencia espectral')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_filter_response(frequencies, magnitude_db, fc, f_max=5000):
    """
    Grafica la respuesta en frecuencia de un filtro.
    
    Parámetros:
    -----------
    frequencies (ndarray): Array de frecuencias
    magnitude_db (ndarray): Respuesta en magnitud (dB)
    fc (float): Frecuencia de corte
    f_max (float): Frecuencia máxima a mostrar
    """
    idx_max = np.where(frequencies >= f_max)[0][0]
    
    plt.figure(figsize=(12, 5))
    plt.plot(frequencies[1:idx_max], magnitude_db[1:idx_max], linewidth=2)
    plt.axvline(x=fc, color='red', linestyle='--', 
               label='Frecuencia de corte')
    plt.axhline(y=-3, color='gray', linestyle=':', 
               label='-3 dB (mitad de potencia)')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Ganancia (dB)')
    plt.title('Respuesta en frecuencia del filtro')
    plt.grid(True)
    plt.legend()
    plt.ylim(-80, 5)
    plt.show()


def plot_impulse_response(filter_coeffs, title="Respuesta al impulso"):
    """Grafica la respuesta al impulso de un filtro"""
    n_taps = len(filter_coeffs)
    n = np.arange(n_taps)
    
    plt.figure(figsize=(12, 5))
    
    # Usar stem para valores discretos
    markerline, stemlines, baseline = plt.stem(n, filter_coeffs)
    plt.setp(markerline, markersize=4)
    
    # Línea horizontal en y=0 para referencia
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    
    # Marcar el centro del filtro
    center = n_taps // 2
    plt.axvline(x=center, color='red', linestyle='--', linewidth=1, 
                alpha=0.5, label=f'Centro (n={center})')
    
    plt.xlabel('n (muestra)')
    plt.ylabel('h[n]')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Ajustar límites Y para ver mejor los detalles
    y_max = np.max(np.abs(filter_coeffs)) * 1.1
    plt.ylim(-y_max, y_max)
    
    plt.show()

def plot_filters_comparison(h_low, h_high, fc, sr):
    """
    Compara filtros paso bajo y paso alto en una sola figura
    Muestra respuestas al impulso y respuestas en frecuencia
    """
    from .analysis import calculate_filter_response
    
    n_taps = len(h_low)
    n = np.arange(n_taps)
    center = n_taps // 2
    
    # Calcular respuestas en frecuencia
    freqs_low, mag_low_db = calculate_filter_response(h_low, sr)
    freqs_high, mag_high_db = calculate_filter_response(h_high, sr)
    
    # Encontrar índice para limitar a 5kHz
    idx_max = np.where(freqs_low >= 5000)[0][0]
    
    # Crear figura con 3 subplots
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 1, hspace=0.3)
    
    # 1. Respuesta al impulso - Lowpass
    ax1 = fig.add_subplot(gs[0])
    ax1.stem(n, h_low, basefmt=' ', label='Paso bajo')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    ax1.axvline(x=center, color='red', linestyle='--', linewidth=1, alpha=0.3)
    ax1.set_ylabel('h_low[n]')
    ax1.set_title(f'Respuesta al impulso - PASO BAJO (fc={fc} Hz)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, n_taps-1)
    
    # 2. Respuesta al impulso - Highpass
    ax2 = fig.add_subplot(gs[1])
    ax2.stem(n, h_high, basefmt=' ', label='Paso alto', linefmt='C1-', markerfmt='C1o')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    ax2.axvline(x=center, color='red', linestyle='--', linewidth=1, alpha=0.3)
    ax2.set_ylabel('h_high[n]')
    ax2.set_xlabel('n (muestra)')
    ax2.set_title(f'Respuesta al impulso - PASO ALTO (fc={fc} Hz)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, n_taps-1)
    
    # 3. Respuestas en frecuencia superpuestas
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(freqs_low[1:idx_max], mag_low_db[1:idx_max], 
             linewidth=2, label='Paso bajo', alpha=0.8)
    ax3.plot(freqs_high[1:idx_max], mag_high_db[1:idx_max], 
             linewidth=2, label='Paso alto', alpha=0.8)
    ax3.axvline(x=fc, color='red', linestyle='--', label=f'fc = {fc} Hz')
    ax3.axhline(y=-3, color='gray', linestyle=':', label='-3 dB', alpha=0.5)
    ax3.set_xlabel('Frecuencia (Hz)')
    ax3.set_ylabel('Ganancia (dB)')
    ax3.set_title('Respuestas en frecuencia - COMPARACIÓN')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='right')
    ax3.set_ylim(-80, 5)
    ax3.set_xlim(0, 5000)
    
    plt.show()

def plot_audio_effects_comparison(freqs, mag_orig_db, mag_low_db, mag_high_db, fc, f_max=5000):
    """
    Compara el efecto de ambos filtros en el audio en una sola figura.
    Muestra original vs ambos filtros en el mismo gráfico.
    """
    idx_max = np.where(freqs >= f_max)[0][0]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # 1. Comparación espectral: Original vs ambos filtros
    axes[0].plot(freqs[1:idx_max], mag_orig_db[1:idx_max],
                 alpha=0.8, label='Original', linewidth=2)
    axes[0].plot(freqs[1:idx_max], mag_low_db[1:idx_max],
                 alpha=0.7, label='Paso bajo', linewidth=1.5)
    axes[0].plot(freqs[1:idx_max], mag_high_db[1:idx_max],
                 alpha=0.7, label='Paso alto', linewidth=1.5)
    axes[0].axvline(x=fc, color='red', linestyle='--', 
                    label=f'Frecuencia de corte ({fc} Hz)', alpha=0.6)
    axes[0].set_ylabel('Magnitud (dB)')
    axes[0].set_title('Comparación espectral - Original vs Filtros')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Diferencia espectral: Ambos filtros
    diferencia_low = mag_orig_db - mag_low_db
    diferencia_high = mag_orig_db - mag_high_db
    
    axes[1].plot(freqs[1:idx_max], diferencia_low[1:idx_max], 
                 label='Atenuación paso bajo', linewidth=1.5)
    axes[1].plot(freqs[1:idx_max], diferencia_high[1:idx_max], 
                 label='Atenuación paso alto', linewidth=1.5)
    axes[1].axvline(x=fc, color='red', linestyle='--', 
                    label=f'Frecuencia de corte ({fc} Hz)', alpha=0.6)
    axes[1].set_xlabel('Frecuencia (Hz)')
    axes[1].set_ylabel('Atenuación (dB)')
    axes[1].set_title('Diferencia espectral - Lo que cada filtro atenuó')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()