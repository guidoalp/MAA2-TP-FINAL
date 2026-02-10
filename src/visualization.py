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
    plt.figure(figsize=(12, 4))
    plt.stem(filter_coeffs)
    plt.xlabel('n (muestra)')
    plt.ylabel('h[n]')
    plt.title(title)
    plt.grid(True)
    plt.show()