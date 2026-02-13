import streamlit as st
import librosa
import numpy as np
import io
from src.filters import lowpass_fir, highpass_fir, bandpass_fir, apply_filter
from src.analysis import calculate_fft, calculate_filter_response, compute_spectrogram

import matplotlib.pyplot as plt
import soundfile as sf
from io import BytesIO


st.set_page_config(
    page_title="Analizador Espectral FIR",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Analizador Espectral con Filtros FIR")
st.markdown("**Trabajo Final - Matemática Aplicada al Arte Digital II**")
st.markdown("---")

st.sidebar.header("⚙️ Configuración")

uploaded_file = st.sidebar.file_uploader(
    "Subir archivo de audio",
    type=['wav', 'mp3'],
    help="Formatos soportados: WAV, MP3"
)

if uploaded_file is not None:
    st.sidebar.success("Archivo cargado!")
    
    audio_bytes = uploaded_file.read()
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
    
    st.sidebar.info(f"""
    **Información del audio:**
    - Frecuencia de muestreo: {sr} Hz
    - Duración: {len(y)/sr:.2f} segundos
    - Muestras: {len(y):,}
    """)
    
    st.sidebar.subheader("👇", text_alignment="center")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Tipo de filtro")
    filter_type = st.sidebar.radio(
        "Seleccionar:",
        ["Paso bajo", "Paso alto", "Paso banda"],
        help="Elige el tipo de filtro a aplicar"
    )
    
    # Parámetros del filtro
    st.sidebar.markdown("---")
    st.sidebar.subheader("Parámetros del filtro")
    
    if filter_type == "Paso banda":
        fc_low = st.sidebar.slider(
            "Frecuencia de corte inferior (Hz)",
            min_value=100,
            max_value=int(sr//2) - 100,
            value=1000,
            step=100,
        )
        fc_high = st.sidebar.slider(
            "Frecuencia de corte superior (Hz)",
            min_value=fc_low + 100,
            max_value=int(sr//2),
            value=3000,
            step=100,
            help="Frecuencia hasta donde pasa"
        )
    else:
        fc = st.sidebar.slider(
            "Frecuencia de corte (Hz)",
            min_value=100,
            max_value=int(sr//2),
            value=3000,
            step=100,
            help="Frecuencia de corte del filtro"
        )
    
    num_taps = st.sidebar.slider(
        "Número de coeficientes (num_taps)",
        min_value=51,
        max_value=501,
        value=101,
        step=50,
        help="Longitud del filtro. Mayor = más preciso"
    )
    
    st.sidebar.markdown("---")
    analyze_button = st.sidebar.button("Analizar", icon="🔍", type="secondary", use_container_width=True)
    
    if analyze_button:
        with st.spinner("Procesando audio..."):
            
            # Diseñar filtro según tipo
            if filter_type == "Paso bajo":
                h = lowpass_fir(fc=fc, fs=sr, num_taps=num_taps)
                filter_name = f"Paso bajo (fc={fc} Hz)"
            elif filter_type == "Paso alto":
                h = highpass_fir(fc=fc, fs=sr, num_taps=num_taps)
                filter_name = f"Paso alto (fc={fc} Hz)"
            else:
                h = bandpass_fir(fc_low=fc_low, fc_high=fc_high, fs=sr, num_taps=num_taps)
                filter_name = f"Paso banda ({fc_low}-{fc_high} Hz)"
            
            y_filtered = apply_filter(y, h)
            
            # Guardar en session_state para usar después
            st.session_state.y = y
            st.session_state.y_filtered = y_filtered
            st.session_state.sr = sr
            st.session_state.h = h
            st.session_state.filter_name = filter_name
            st.session_state.filter_type = filter_type
            if filter_type == "Paso banda":
                st.session_state.fc_low = fc_low
                st.session_state.fc_high = fc_high
            else:
                st.session_state.fc = fc
            
            st.success("Análisis completado!")
            st.rerun()

else:
    st.info("👈 Por favor, subí un archivo de audio para comenzar")
    st.stop()

# Si ya se analizó, muestro resultados
if 'y_filtered' in st.session_state:
    st.markdown("---")
    st.header("📊 Resultados del Análisis")
    
    # Recuperar datos del session_state
    y = st.session_state.y
    y_filtered = st.session_state.y_filtered
    sr = st.session_state.sr
    h = st.session_state.h
    filter_name = st.session_state.filter_name
    filter_type = st.session_state.filter_type
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Audio y Forma de Onda",
        "Diseño del Filtro",
        "Análisis Espectral (FFT)",
        "Espectrogramas (STFT)"
    ])
    
    # ============================================
    # Audio y forma de onda
    # ============================================
    with tab1:
        st.subheader("Forma de onda")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original**")
            buffer_orig = BytesIO()
            sf.write(buffer_orig, y, sr, format='WAV')
            buffer_orig.seek(0)
            st.audio(buffer_orig, format='audio/wav')

            fig, ax = plt.subplots(figsize=(10, 3))
            tiempo = np.linspace(0, len(y)/sr, len(y))
            ax.plot(tiempo, y, linewidth=0.5)
            ax.set_xlabel('Tiempo (s)')
            ax.set_ylabel('Amplitud')
            ax.set_title('Señal original')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown(f"**Filtrado - {filter_name}**")
            buffer_filt = BytesIO()
            sf.write(buffer_filt, y_filtered, sr, format='WAV')
            buffer_filt.seek(0)
            st.audio(buffer_filt, format='audio/wav')

            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(tiempo, y_filtered, linewidth=0.5, color='orange')
            ax.set_xlabel('Tiempo (s)')
            ax.set_ylabel('Amplitud')
            ax.set_title('Señal filtrada')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
    
    # ============================================
    # Diseño de Filtro
    # ============================================
    with tab2:
        st.subheader("Respuesta al impulso y en frecuencia")
        
        # Respuesta al impulso
        st.markdown("**Respuesta al impulso h[n]**")
        fig, ax = plt.subplots(figsize=(12, 4))
        n = np.arange(len(h))
        center = len(h) // 2
        ax.stem(n, h, basefmt=' ')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
        ax.axvline(x=center, color='red', linestyle='--', linewidth=1, alpha=0.5,
                  label=f'Centro (n={center})')
        ax.set_xlabel('n (muestra)')
        ax.set_ylabel('h[n]')
        ax.set_title(f'Respuesta al impulso - {filter_name}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        plt.close()
        
        # Respuesta en frecuencia
        st.markdown("**Respuesta en frecuencia H(f)**")
        freqs_filter, mag_filter_db = calculate_filter_response(h, sr)
        idx_max = np.where(freqs_filter >= 10000)[0][0]
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(freqs_filter[1:idx_max], mag_filter_db[1:idx_max], linewidth=2)
        
        # Marcar frecuencias de corte según tipo de filtro
        if filter_type == "Paso banda":
            fc_low = st.session_state.fc_low
            fc_high = st.session_state.fc_high
            ax.axvline(x=fc_low, color='red', linestyle='--', 
                      label=f'fc_low = {fc_low} Hz')
            ax.axvline(x=fc_high, color='red', linestyle='--', 
                      label=f'fc_high = {fc_high} Hz')
        else:
            fc = st.session_state.fc
            ax.axvline(x=fc, color='red', linestyle='--', 
                      label=f'fc = {fc} Hz')
        
        ax.axhline(y=-3, color='gray', linestyle=':', label='-3 dB', alpha=0.5)
        ax.set_xlabel('Frecuencia (Hz)')
        ax.set_ylabel('Ganancia (dB)')
        ax.set_title(f'Respuesta en frecuencia - {filter_name}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(-80, 5)
        ax.set_xlim(0, 10000)
        st.pyplot(fig)
        plt.close()
    
    # ============================================
    # Análisis Espectral (FFT)
    # ============================================
    with tab3:
        st.subheader("Comparación espectral (FFT)")
        
        # Calcular FFT
        freqs_orig, _, mag_orig_db = calculate_fft(y, sr)
        freqs_filt, _, mag_filt_db = calculate_fft(y_filtered, sr)
        
        idx_max = np.where(freqs_orig >= 10000)[0][0]
        
        # Gráfico de comparación
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # Espectros superpuestos
        axes[0].plot(freqs_orig[1:idx_max], mag_orig_db[1:idx_max],
                    alpha=0.7, label='Original', linewidth=1.5)
        axes[0].plot(freqs_filt[1:idx_max], mag_filt_db[1:idx_max],
                    alpha=0.7, label='Filtrado', linewidth=1.5)
        
        # Marcar frecuencias de corte
        if filter_type == "Paso banda":
            axes[0].axvline(x=fc_low, color='red', linestyle='--', alpha=0.6)
            axes[0].axvline(x=fc_high, color='red', linestyle='--', alpha=0.6)
        else:
            axes[0].axvline(x=fc, color='red', linestyle='--', 
                          label='Frecuencia de corte', alpha=0.6)
        
        axes[0].set_ylabel('Magnitud (dB)')
        axes[0].set_title('Comparación espectral')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Diferencia espectral
        diferencia_db = mag_orig_db - mag_filt_db
        axes[1].plot(freqs_orig[1:idx_max], diferencia_db[1:idx_max], 
                    color='green', linewidth=1.5)
        
        if filter_type == "Paso banda":
            axes[1].axvline(x=fc_low, color='red', linestyle='--', alpha=0.6)
            axes[1].axvline(x=fc_high, color='red', linestyle='--', alpha=0.6)
        else:
            axes[1].axvline(x=fc, color='red', linestyle='--', 
                          label='Frecuencia de corte', alpha=0.6)
        
        axes[1].set_xlabel('Frecuencia (Hz)')
        axes[1].set_ylabel('Atenuación (dB)')
        axes[1].set_title('Diferencia espectral (lo que se atenuó)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # ============================================
    # Espectrogramas (STFT)
    # ============================================
    with tab4:
        st.subheader("Espectrogramas (análisis tiempo-frecuencia)")
        
        # Calcular espectrogramas
        times_orig, freqs_spec, _, Sxx_orig_db = compute_spectrogram(y, sr)
        times_filt, _, _, Sxx_filt_db = compute_spectrogram(y_filtered, sr)
        
        # Limitar a 10kHz
        idx_freq = np.where(freqs_spec <= 10000)[0]
        
        # Rango común de colores
        vmin = min(Sxx_orig_db.min(), Sxx_filt_db.min())
        vmax = max(Sxx_orig_db.max(), Sxx_filt_db.max())
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Original
        im1 = axes[0].pcolormesh(times_orig, freqs_spec[idx_freq], 
                                 Sxx_orig_db[idx_freq, :],
                                 shading='gouraud', cmap='viridis', 
                                 vmin=vmin, vmax=vmax)
        axes[0].set_ylabel('Frecuencia (Hz)')
        axes[0].set_title('Espectrograma - Audio original')
        axes[0].set_ylim(0, 10000)
        fig.colorbar(im1, ax=axes[0], label='Magnitud (dB)')
        
        # Filtrado
        im2 = axes[1].pcolormesh(times_filt, freqs_spec[idx_freq], 
                                 Sxx_filt_db[idx_freq, :],
                                 shading='gouraud', cmap='viridis', 
                                 vmin=vmin, vmax=vmax)
        
        # Marcar frecuencias de corte
        if filter_type == "Paso banda":
            axes[1].axhline(y=fc_low, color='red', linestyle='--', 
                          linewidth=2, alpha=0.7)
            axes[1].axhline(y=fc_high, color='red', linestyle='--', 
                          linewidth=2, alpha=0.7)
        else:
            axes[1].axhline(y=fc, color='red', linestyle='--', 
                          linewidth=2, label=f'fc = {fc} Hz', alpha=0.7)
            axes[1].legend(loc='upper right')
        
        axes[1].set_xlabel('Tiempo (s)')
        axes[1].set_ylabel('Frecuencia (Hz)')
        axes[1].set_title(f'Espectrograma - {filter_name}')
        axes[1].set_ylim(0, 10000)
        fig.colorbar(im2, ax=axes[1], label='Magnitud (dB)')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ============================================
    # DESCARGA
    # ============================================
    st.markdown("---")
    st.header("💾 Descargar audios")

    original_filename = uploaded_file.name.rsplit('.', 1)[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Audio original**")
        
        # Preparar archivo para descarga
        buffer_orig_download = BytesIO()
        sf.write(buffer_orig_download, y, sr, format='WAV')
        buffer_orig_download.seek(0)
        
        st.download_button(
            label="Descargar original (.wav)",
            data=buffer_orig_download,
            file_name="audio_original.wav",
            mime="audio/wav",
            use_container_width=True
        )
    
    with col2:
        st.markdown(f"**Audio filtrado - {filter_name}**")
        
        # Preparar archivo para descarga
        buffer_filt_download = BytesIO()
        sf.write(buffer_filt_download, y_filtered, sr, format='WAV')
        buffer_filt_download.seek(0)
        
        # Generar nombre de archivo descriptivo
        if filter_type == "Paso banda":
            filename = f"{original_filename}_{fc_low}-{fc_high}Hz.wav"
        else:
            filter_prefix = "lowpass" if filter_type == "Paso bajo" else "highpass"
            filename = f"{original_filename}_{filter_prefix}_{fc}Hz.wav"
        
        st.download_button(
            label="Descargar filtrado (.wav)",
            data=buffer_filt_download,
            file_name=filename,
            mime="audio/wav",
            use_container_width=True
        )
else:
    st.info("Presiona el botón 'Analizar' para ver los resultados")