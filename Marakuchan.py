import pyttsx3
import speech_recognition as sr
import numpy as np
import datetime
import webbrowser
import queue
import threading
import sys
import time
import pygame
import pyjokes
import requests
import cv2
import numpy as np
from PIL import ImageGrab
import pyautogui
from collections import deque
from datetime import datetime
import os
from threading import Lock
import sounddevice as sd
import wave
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip

# Nombre de activación del asistente
NOMBRE_ASISTENTE = "maraku-chan"

# Configuración de grabación
RECORDING_FPS = 30  # FPS de la grabacón
BUFFER_SECONDS = 300  # 5 minutos
BUFFER_SIZE = RECORDING_FPS * BUFFER_SECONDS
OUTPUT_DIRECTORY = "clips"
AUDIO_SAMPLERATE = 44100
AUDIO_CHANNELS = 2

# Configuración de Mixtral AI
MODEL_ID = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
API_URL = f'https://api-inference.huggingface.co/models/{MODEL_ID}'
API_TOKEN = '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'

headers = {
    'Authorization': f'Bearer {API_TOKEN}',
    'Content-Type': 'application/json'
}

# Configuración de Pygame
pygame.init()
WINDOW_WIDTH = 700
WINDOW_HEIGHT = 700
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Maraku-chan")


# Asegurarse de que existe el directorio para clips
if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)

class ScreenRecorder:
    def __init__(self):
        self.frame_buffer = deque(maxlen=BUFFER_SIZE)
        self.audio_buffer = deque(maxlen=AUDIO_SAMPLERATE * BUFFER_SECONDS * AUDIO_CHANNELS)
        self.recording = False
        self.recording_thread = None
        self.audio_thread = None
        self.buffer_lock = Lock()
        self.audio_lock = Lock()
        
    def start_recording(self):
        """Iniciar la grabación en segundo plano"""
        if not self.recording:
            self.recording = True
            self.recording_thread = threading.Thread(target=self._record_screen, daemon=True)
            self.audio_thread = threading.Thread(target=self._record_audio, daemon=True)
            self.recording_thread.start()
            self.audio_thread.start()
    
    def _record_screen(self):
        """Grabar la pantalla continuamente"""
        while self.recording:
            try:
                screenshot = pyautogui.screenshot()
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                with self.buffer_lock:
                    self.frame_buffer.append(frame)
                time.sleep(1/RECORDING_FPS)
            except Exception as e:
                print(f"Error durante la grabación de video: {e}")
                continue

    def _record_audio(self):
        """Grabar el audio del sistema continuamente"""
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Error de audio: {status}")
            with self.audio_lock:
                # Convertir a int16 para compatibilidad con wave
                audio_data = (indata * 32767).astype(np.int16)
                self.audio_buffer.extend(audio_data.flatten())

        try:
            with sd.InputStream(channels=AUDIO_CHANNELS, 
                              samplerate=AUDIO_SAMPLERATE,
                              dtype=np.float32,
                              callback=audio_callback):
                while self.recording:
                    sd.sleep(100)
        except Exception as e:
            print(f"Error durante la grabación de audio: {e}")
    
    def save_clip(self):
        """Guardar los últimos 5 minutos como un video con audio"""
        try:
            # Guardar frames
            with self.buffer_lock:
                if len(self.frame_buffer) == 0:
                    return "No hay frames grabados para guardar"
                frames_to_save = list(self.frame_buffer)
            
            # Guardar audio
            with self.audio_lock:
                audio_to_save = np.array(list(self.audio_buffer), dtype=np.int16)
            
            if not frames_to_save:
                return "No hay frames para guardar"
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_video_path = os.path.join(OUTPUT_DIRECTORY, f"temp_video_{timestamp}.mp4")
            temp_audio_path = os.path.join(OUTPUT_DIRECTORY, f"temp_audio_{timestamp}.wav")
            final_output_path = os.path.join(OUTPUT_DIRECTORY, f"clip_{timestamp}.mp4")
            
            # Guardar video temporal
            height, width = frames_to_save[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, RECORDING_FPS, (width, height))
            
            for frame in frames_to_save:
                out.write(frame)
            out.release()
            
            # Guardar audio temporal usando wave
            with wave.open(temp_audio_path, 'wb') as wav_file:
                wav_file.setnchannels(AUDIO_CHANNELS)
                wav_file.setsampwidth(2)  # 2 bytes para int16
                wav_file.setframerate(AUDIO_SAMPLERATE)
                wav_file.writeframes(audio_to_save.tobytes())
            
            try:
                # Combinar video y audio
                video_clip = VideoFileClip(temp_video_path)
                audio_clip = AudioFileClip(temp_audio_path)
                
                # Asegurar que el audio tenga la misma duración que el video
                audio_clip = audio_clip.subclip(0, video_clip.duration)
                
                final_clip = video_clip.set_audio(audio_clip)
                final_clip.write_videofile(final_output_path, 
                                         codec='libx264', 
                                         audio_codec='aac',
                                         fps=RECORDING_FPS)
                
                # Limpiar archivos temporales
                video_clip.close()
                audio_clip.close()
                os.remove(temp_video_path)
                os.remove(temp_audio_path)
                
                return final_output_path
            except Exception as e:
                print(f"Error al procesar el video final: {e}")
                # Si falla la combinación, al menos guardamos el video sin audio
                if os.path.exists(temp_video_path):
                    final_path = os.path.join(OUTPUT_DIRECTORY, f"clip_video_only_{timestamp}.mp4")
                    os.rename(temp_video_path, final_path)
                    return final_path
                
        except Exception as e:
            print(f"Error al guardar el clip: {e}")
            return f"Error al guardar el clip: {str(e)}"

    def stop_recording(self):
        """Detener la grabación"""
        self.recording = False
        if self.recording_thread:
            self.recording_thread.join()
        if self.audio_thread:
            self.audio_thread.join()

class TalkingCharacter:
    def __init__(self):
        # Cargar las imágenes del personaje
        self.images = [
            pygame.image.load('marakuchanPersonaje/MARAKU-ARUSIO.png'),
            pygame.image.load('marakuchanPersonaje/MARAKU-ARUSIO-MEDIO.png'),
            pygame.image.load('marakuchanPersonaje/MARAKU-ARUSIO-ABIERTA.png')
        ]
        
        # Redimensionar las imágenes
        self.images = [pygame.transform.scale(img, (700, 700)) for img in self.images]
        
        self.current_image = 0
        self.is_talking = False
        self.talk_timer = 0
        self.talk_delay = 100  # Milisegundos entre cambios de imagen
        
        self.x = WINDOW_WIDTH // 2 - 350
        self.y = WINDOW_HEIGHT // 2 - 350

    def start_talking(self):
        self.is_talking = True
        self.talk_timer = pygame.time.get_ticks()

    def stop_talking(self):
        self.is_talking = False
        self.current_image = 0

    def update(self):
        if self.is_talking:
            current_time = pygame.time.get_ticks()
            if current_time - self.talk_timer > self.talk_delay:
                self.current_image = (self.current_image + 1) % len(self.images)
                self.talk_timer = current_time

    def draw(self, surface):
        surface.blit(self.images[self.current_image], (self.x, self.y))

# Crear instancia del personaje
character = TalkingCharacter()

# Variables de control
is_speaking = threading.Event()
is_listening = threading.Event()
is_listening.set()  # Inicialmente está escuchando

# Inicializar el motor de síntesis de voz
engine = pyttsx3.init(driverName='sapi5')

# Configurar el idioma y la velocidad de la voz
engine.setProperty('rate', 150)
voices = engine.getProperty('voices')
spanish_voice = next((voice for voice in voices if 'spanish' in voice.name.lower()), None)
if spanish_voice:
    engine.setProperty('voice', spanish_voice.id)
else:
    print("No se encontró una voz en español. Usando la voz predeterminada.")

# Cola para comunicación entre hilos
texto_queue = queue.Queue()

def update_window():
    """Función para actualizar la ventana de Pygame"""
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        character.update()
        window.fill((255, 255, 255))
        character.draw(window)
        pygame.display.flip()
        pygame.time.Clock().tick(60)

def hablar(texto):
    """Función para hacer que el asistente hable."""
    is_speaking.set()
    is_listening.clear()  # Pausar la escucha mientras habla
    print(f"Asistente dice: {texto}")
    
    character.start_talking()
    engine.say(texto)
    engine.runAndWait()
    character.stop_talking()
    
    is_speaking.clear()
    is_listening.set()  # Reanudar la escucha

def consultar_informacion(pregunta):
    """Función para realizar consultas a la API de Mixtral AI."""
    data = {
        "inputs": pregunta,
        "parameters": {
            "max_new_tokens": 250,
            "temperature": 0.7,
            "return_full_text": False
        }
    }

    response = requests.post(API_URL, headers=headers, json=data)
    
    if response.status_code == 200:
        respuesta = response.json()[0]['generated_text']
        hablar(respuesta)
        return respuesta
    else:
        hablar("Lo siento, no pude obtener una respuesta en este momento.")
        return "Error en la consulta"

def escuchar_continuo():
    """Función para escuchar continuamente."""
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.energy_threshold = 2000
    
    with sr.Microphone() as source:
        print("Ajustando al ruido ambiental...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        while True:
            if not is_listening.is_set():
                time.sleep(0.1)  # Esperar si no está escuchando
                continue
                
            try:
                print("\nEscuchando...")
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=5)
                try:
                    texto = recognizer.recognize_google(audio, language="es-ES").lower()
                    print(f"Reconocido: {texto}")
                    texto_queue.put(texto)
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as e:
                    print(f"Error con el servicio de reconocimiento: {e}")
                    continue
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                print(f"Error inesperado: {e}")
                continue

screen_recorder = ScreenRecorder()

def procesar_comando(comando):
    """Función para procesar los comandos de voz."""
    print(f"Procesando comando: {comando}")
    
    if "clipea eso" in comando or "guarda clip" in comando:
        hablar("Guardando los últimos 5 minutos de grabación")
        try:
            output_path = screen_recorder.save_clip()
            if output_path.startswith("Error"):
                hablar("Lo siento, hubo un error al guardar el clip")
            else:
                hablar(f"Clip guardado en {output_path}")
        except Exception as e:
            hablar("Lo siento, hubo un error al guardar el clip")
            print(f"Error al guardar clip: {e}")
        return "desactivar"
    elif "funciones" in comando:
        hablar(f"Mis comandos son los siguientes: Clipear, dar la hora, dar clima, buscar en google, generar textos o poemas, investigar y contar chistes. Tambien puedes despedirte diciendome adios.")
        return "desactivar"
    elif "hora" in comando or "fecha" in comando:
        fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M")
        hablar(f"La fecha y hora actual es {fecha_hora}")

    elif "clima" in comando:
        # Extraer la ciudad del comando
        ciudad = None
        if "de " in comando:
            ciudad = comando.split("de ", 1)[1].strip()
        
        if not ciudad:
            hablar("¿De qué ciudad quieres saber el clima?")
            return "esperar_ciudad"
            
        return consultar_clima(ciudad)

    elif "busca en google" in comando or "buscar información sobre" in comando:
        busqueda = comando.replace("busca en google", "").replace("buscar información sobre", "").strip()
        if busqueda:
            url = f"https://www.google.com/search?q={busqueda}"
            webbrowser.open(url)
            hablar(f"He buscado {busqueda} en Google")
        else:
            hablar("¿Qué quieres que busque?")
            return "esperar_busqueda"

    elif "genera" in comando or "crea un poema sobre" in comando:
        hablar("Dame un momento")
        prompt = f"""{comando} de 250 caracteres (trata de terminar las frases),
        Si vas a hacer guiones, agrega nombre a los personajes,
        evita usar simbolos y se conciso"""
        consultar_informacion(prompt)

    elif "investiga" in comando:
        pregunta = f"""Responde la siguiente pregunta y exclusivamente esta pregunta:{comando.replace("investiga sobre", "").strip()}, 
        responde obligatoriamente en 250 caracteres (trata de terminar las frases),
        no uses palabras con connotaciones negativas,
        no uses palabras con connotaciones positivas,
        no uses palabras con doble sentido,
        no uses palabras con un significado distinto al habitual,
        no uses palabras con un significado distinto al que estamos hablando,
        no uses palabras con un significado distinto al que estamos hablando en este contexto,
        no uses palabras con un significado distinto al que estamos hablando en este contexto y momento 
        evita usar simbolos, evita citar otras personas, 
        evita decir cosas incoherentes y se conciso"""
        hablar("Dame un segundo.")
        consultar_informacion(pregunta)

    elif "chiste" in comando:
        hablar("Espero que te encante mi chiste:")
        hablar(pyjokes.get_joke(language='es', category='all'))

    elif "adiós" in comando:
        hablar("Hasta luego, que tengas un buen día")
        screen_recorder.stop_recording()
        return False
    else:
        hablar("No he entendido el comando. Por favor, repítelo.")
        return "repetir"

    return "desactivar"

def consultar_clima(ciudad):
    """Función para consultar el clima de una ciudad específica."""
    api_key = "b175155b402c44d69e3a12377008afca"
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "appid=" + api_key + "&q=" + ciudad
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404" and x["cod"] != "400":
        y = x["main"]
        current_temperature = str(round(float(y["temp"] - 273.15), 0))
        current_pressure = y["pressure"]
        current_humidity = y["humidity"]
        hablar(f"La temperatura en {ciudad} es {current_temperature} grados centígrados, la presión atmosférica es {current_pressure} hPa, y la humedad es {current_humidity} por ciento")
    else:
        hablar("La ciudad no fue encontrada o no la entendí")
    
    return "desactivar"

def main():
    """Función principal del asistente de voz."""
    # Iniciar el hilo de actualización de la ventana
    threading.Thread(target=update_window, daemon=True).start()
    
    # Iniciar el hilo de escucha
    thread_escucha = threading.Thread(target=escuchar_continuo, daemon=True)
    thread_escucha.start()
    
    # Iniciar la grabación de pantalla
    screen_recorder.start_recording()
    
    hablar(f"Hola, soy {NOMBRE_ASISTENTE}. Puedes activarme diciendo {NOMBRE_ASISTENTE}.")
    
    try:
        estado = "inactivo"
        esperando_ciudad = False
        
        while True:
            try:
                texto = texto_queue.get(timeout=0.1)
                
                if estado == "inactivo" and NOMBRE_ASISTENTE in texto or "maracucho" in texto or "maracucha" in texto:
                    estado = "activo"
                    esperando_ciudad = False
                    hablar("Dime. ¿En qué puedo ayudarte?")
                elif estado == "activo":
                    if esperando_ciudad:
                        resultado = consultar_clima(texto)
                        esperando_ciudad = False
                    else:
                        resultado = procesar_comando(texto)
                        if resultado == "esperar_ciudad":
                            esperando_ciudad = True
                            continue
                    
                    if resultado == "repetir":
                        continue
                    elif resultado == "desactivar":
                        estado = "inactivo"
                        esperando_ciudad = False
                        print("Asistente esperando ser llamado nuevamente.")
                    elif not resultado:
                        break
                    
            except queue.Empty:
                continue
                
    except KeyboardInterrupt:
        print("\nPrograma terminado por el usuario.")
        screen_recorder.stop_recording()
    except Exception as e:
        print(f"Error: {e}")
        screen_recorder.stop_recording()
    finally:
        print("Asistente virtual detenido.")
        pygame.quit()
        
if __name__ == "__main__":
    main()