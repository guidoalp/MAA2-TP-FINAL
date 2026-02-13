import numpy as np

def lowpass_fir(fc, fs, num_taps=101, window_type='hamming'):
	"""
	Filtro FIR pasa bajo que utiliza el método de ventana

	Parámetros
	----------
	- fc (float): Frecuencia de corte en Hz
	- fs (float): Frecuencia de muestreo en Hz
	- num_taps (int): Longitud del filtro (impar)
	- window_type (str): Tipo de ventana: 'hamming', 'blackman', 'hann', 'rectangular'

	Devuelve
	----------
	- h (ndarray): Respuesta al impulso del filtro
	"""

  	# Forzar num_taps impar
	if num_taps % 2 == 0:
		num_taps += 1
		print(f"Advertencia: num_taps ajustado a {num_taps} (debe ser impar)")

	# Normalizar frecuencia de corte
	fc_norm = fc / fs

	# Crear indices centrados en 0
	n = np.arange(num_taps)
	n = n - (num_taps - 1) / 2

	# sinc
	h = np.sinc(2 * fc_norm * n)

	# ventana
	if window_type == 'hamming':
		window = np.hamming(num_taps)
	elif window_type == 'blackman':
		window = np.blackman(num_taps)
	elif window_type == 'hann':
		window = np.hanning(num_taps)
	elif window_type == 'rectangular':
		window = np.ones(num_taps)
	else:
		raise ValueError(f"Ventana '{window_type}' no reconocida")

	h = h * window

	# Normalizo para que la suma de los coeficientes sea = 1 y evitar modificar las frecuencias  que pasan
	h = h / np.sum(h)

	return h

def highpass_fir(fc, fs, num_taps=101, window_type='hamming'):
	"""
	Filtro FIR pasa alto que utiliza el método de ventana e inversión espectral

	Parámetros
	----------
	- fc (float): Frecuencia de corte en Hz
	- fs (float): Frecuencia de muestreo en Hz
	- num_taps (int): Longitud del filtro (impar)
	- window_type (str): Tipo de ventana: 'hamming', 'blackman', 'hann', 'rectangular'

	Devuelve
	----------
	- h (ndarray): Respuesta al impulso del filtro
	"""
	# Forzar num_taps impar
	if num_taps % 2 == 0:
		num_taps += 1
		print(f"Advertencia: num_taps ajustado a {num_taps} (debe ser impar)")
	
	h_lowpass = lowpass_fir(fc, fs, num_taps, window_type)

	impulso = np.zeros(num_taps)
	impulso[num_taps // 2] = 1

	h_high = impulso - h_lowpass

	return h_high

def bandpass_fir(fc_low, fc_high, fs, num_taps=101, window_type='hamming'):
    """
    Filtro FIR paso banda usando el método de ventana. Combina paso alto y paso bajo.
    
    Parámetros:
    -----------
    fc_low (float): Frecuencia de corte inferior en Hz
    fc_high (float): Frecuencia de corte superior en Hz
    fs (float): Frecuencia de muestreo en Hz
    num_taps (int): Longitud del filtro (debe ser impar)
    window_type (str): Tipo de ventana: 'hamming', 'blackman', 'hann', 'rectangular'
    
    Retorna:
    --------
    h (ndarray): Respuesta al impulso del filtro paso banda
    """
    # validación de parametros
    if fc_low >= fc_high:
        print("Error: fc_low debe ser menor que fc_high")
        return None
    
    if num_taps % 2 == 0:
        num_taps += 1
    
    h_low = lowpass_fir(fc_high, fs, num_taps, window_type)
    
    h_high = highpass_fir(fc_low, fs, num_taps, window_type)
    
    # convolución de ambos filtros
    h_bp = np.convolve(h_low, h_high, mode='same')
    
    # normalizar
    h_bp = h_bp / np.sum(np.abs(h_bp))
    
    return h_bp

def apply_filter(signal, filter_coeffs):
	"""
	Aplica un filtro FIR a una señal usando convolución

	Parámetros
	----------
	- signal (ndarray): Señal de entrada
	- filter_coeffs (ndarray): Coeficientes del filtro

	Devuelve
	----------
	- filtered_signal (ndarray): Señal filtrada (misma longitud que la entrada)
	"""
	return np.convolve(signal, filter_coeffs, mode='same')