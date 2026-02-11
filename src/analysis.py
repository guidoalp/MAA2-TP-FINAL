import numpy as np

def calculate_fft(signal, sr):
    """
    Calcula la FFT de la señal y retorna magnitud y frecuencias
    
    Parametros
    ----------
    - signal (ndarray): Señal en el dominio del tiempo
    - sr (float): Frecuencia de muestreo
    
    Retorna
    ----------
    - frequencies (ndarray): Array de frecuencias (Hz)
    - magnitude (ndarray): Magnitud del espectro
    - magnitude_db (ndarray): Magnitud en dB
    """
    fft_result = np.fft.fft(signal)
    magnitude = np.abs(fft_result)
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    frequencies = np.fft.fftfreq(len(signal), 1/sr)
    
    return frequencies, magnitude, magnitude_db


def calculate_filter_response(filter_coeffs, sr, nfft=2048):
    """
    Calcula la respuesta en frecuencia de un filtro.
    
    Parametros
    ----------
    - filter_coeffs (ndarray): Coeficientes del filtro
    - sr (float): Frecuencia de muestreo
    - nfft (int): Número de puntos para la FFT (mayor = más resolución)
    
    Retorna
    ----------
    - frequencies (ndarray): Array de frecuencias
    - magnitude_db (ndarray): Respuesta en magnitud (dB)
    """
    fft_filter = np.fft.fft(filter_coeffs, n=nfft)
    magnitude = np.abs(fft_filter)
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    frequencies = np.fft.fftfreq(nfft, 1/sr)
    
    return frequencies, magnitude_db


def spectral_difference(signal_original, signal_processed, sr):
    """
    Calcula la diferencia espectral entre dos señales.
    
    Parámetros:
    -----------
    - signal_original (ndarray): Señal original
    - signal_processed (ndarray): Señal procesada
    - sr (float): Frecuencia de muestreo

    Retorna:
    --------
    - frequencies (ndarray): Array de frecuencias
    - difference_db (ndarray): Diferencia en dB (original - procesada)
    """
    freqs_orig, _, magnitude_orig_db = calculate_fft(signal_original, sr)
    _, _, magnitude_proccesed_db = calculate_fft(signal_processed, sr)
    
    difference_db = magnitude_orig_db - magnitude_proccesed_db
    
    return freqs_orig, difference_db

def compute_spectrogram(signal, sr, nperseg=2048, noverlap=None):
    """
    Calcula el espectrograma de una señal usando STFT.
    
    Parámetros:
    -----------
    - signal (ndarray): Señal en el dominio del tiempo
    - sr (float): Frecuencia de muestreo
    - nperseg (int): Longitud de cada segmento (ventana). Default: 2048
    - noverlap (int, optional): Número de muestras de solapamiento. Default: nperseg // 2

    Retorna:
    --------
    - times (ndarray): Array de tiempos (segundos)
    - frequencies (ndarray): Array de frecuencias (Hz)
    - Sxx (ndarray): Espectrograma (magnitud al cuadrado)
    - Sxx_db (ndarray): Espectrograma en dB
    """
    from scipy import signal as scipy_signal
    
    if noverlap is None:
        noverlap = nperseg // 2
    
    frequencies, times, Sxx = scipy_signal.spectrogram(
        signal,
        fs=sr,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density'
    )
    
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    return times, frequencies, Sxx, Sxx_db