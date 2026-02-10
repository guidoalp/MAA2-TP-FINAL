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

	# Calculo la sinc
	h = np.sinc(2 * fc_norm * n)

	# Aplico ventana
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