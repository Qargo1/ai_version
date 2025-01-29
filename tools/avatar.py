'''
Дополнительные улучшения:
LOD (Levels of Detail) :
В Three.js можно использовать уровни детализации для оптимизации рендеринга моделей 1.
Например, можно создать несколько версий модели с разной степенью детализации и переключаться 
между ними в зависимости от расстояния до камеры.
Кеширование анимаций :
Можно сохранять результаты анализа аудио и анимаций в кеш, чтобы не повторять одни и те же 
вычисления при каждом запуске программы.
Использование WebAssembly :
WebAssembly может быть использован для выполнения сложных вычислений на стороне клиента, 
что позволит ускорить обработку данных 7.
Например, можно использовать Pyodide для выполнения Python-кода в браузере через WebAssembly.
Fallback на 2D аватар :
Если VRAM недостаточно для загрузки 3D модели, можно реализовать отображение 2D аватара как 
альтернативное решение.
Это может быть сделано путем проверки доступного VRAM и переключения на 2D режим при 
необходимости.
Управление освещением и система жестов :
Для управления освещением можно использовать API Three.js для изменения параметров источников 
света и материалов объектов.
Система жестов может быть реализована с помощью компьютерного зрения и технологий 
распознавания движений, таких как OpenPose или MediaPipe.
Поддержка VR гарнитур :
Three.js поддерживает работу с VR гарнитурами через специальные плагины и API, такие как 
WebXR Device API 1.
'''


import os
import urllib.error
import json
import threading
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from urllib.request import urlretrieve
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtWebEngineWidgets import QWebEngineView
from scipy.fft import fft  # Используем библиотеку scipy для FFT
import logging  # Логирование

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Путь к модели по умолчанию
DEFAULT_MODEL_PATH = "models/visual/alleyana.glb"

@dataclass
class AvatarConfig:
    model_url: str
    camera_fov: int = 45
    animation_speed: float = 1.0
    emotion: str = "neutral"
    viseme_map: Dict[str, float] = None


class AvatarController:
    def __init__(self, config: AvatarConfig):
        self.config = config
        self.local_model_path = DEFAULT_MODEL_PATH
        self.use_3d_avatar = True  # Флаг для использования 3D-аватара
        self._init_emotion_map()
        self.viewer = None
        self.animation_thread = None
        self._load_assets()
        self._init_web_view()

    def _init_emotion_map(self):
        """Инициализация карты эмоций по умолчанию"""
        if not self.config.viseme_map:
            self.config.viseme_map = {
                'sil': 0.0, 'PP': 0.3, 'FF': 0.2, 'TH': 0.4,
                'DD': 0.35, 'kk': 0.5, 'CH': 0.4, 'SS': 0.3,
                'nn': 0.25, 'RR': 0.4, 'aa': 0.6, 'E': 0.5
            }

    def _load_assets(self):
        """Кеширование 3D модели локально"""
        cache_dir = os.path.join(os.path.expanduser("~"), ".avatar_cache")
        os.makedirs(cache_dir, exist_ok=True)
        filename = os.path.basename(self.config.model_url)
        self.local_model_path = os.path.join(cache_dir, filename)
        try:
            logging.info(f"Downloading model from: {self.config.model_url}")
            urlretrieve(self.config.model_url, self.local_model_path)
            logging.info("Model downloaded successfully.")
        except urllib.error.HTTPError as e:
            logging.warning(f"HTTP Error: {e}. Using fallback model.")
            self.local_model_path = DEFAULT_MODEL_PATH
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            self.local_model_path = DEFAULT_MODEL_PATH

        # Проверка существования файла после загрузки
        if not os.path.exists(self.local_model_path):
            logging.error("Model file does not exist after download. Falling back to default.")
            self.local_model_path = DEFAULT_MODEL_PATH
            self.use_3d_avatar = False  # Отключаем 3D-аватар при ошибке

    def _init_web_view(self):
        """Инициализация WebGL вьювера"""
        self.app = QApplication.instance() or QApplication([])
        self.container = QWidget()
        self.viewer = QWebEngineView(self.container)
        self._load_html_template()
        self.container.show()

    def _load_html_template(self):
        """Загрузка HTML шаблона с Three.js"""
        if not self.use_3d_avatar:
            logging.warning("3D avatar is disabled due to errors.")
            return

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
            <style>
                body {{ margin: 0; }}
                canvas {{ width: 100% !important; height: 100% !important; }}
            </style>
        </head>
        <body>
            <script>
                let mixer, avatar;
                const visemes = {json.dumps(self.config.viseme_map)};
                
                // Инициализация сцены
                const scene = new THREE.Scene();
                const camera = new THREE.PerspectiveCamera({self.config.camera_fov}, window.innerWidth/window.innerHeight, 0.1, 1000);
                const renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(window.innerWidth, window.innerHeight);
                document.body.appendChild(renderer.domElement);
                
                // Загрузка модели
                new THREE.GLTFLoader().load(
                    'file://{self.local_model_path}',
                    function(gltf) {{
                        avatar = gltf.scene;
                        scene.add(avatar);
                        camera.position.set(0, 1.6, 2);
                        new THREE.OrbitControls(camera, renderer.domElement);
                    }},
                    undefined,
                    function(error) {{
                        console.error('Failed to load model:', error);
                        alert('Failed to load 3D model. Fallback to 2D avatar.');
                        // Здесь можно добавить отображение 2D аватара
                    }}
                );
                
                // Анимация
                function animateViseme(phoneme) {{
                    if (!avatar) return;
                    avatar.traverse(child => {{
                        if (child.morphTargetInfluences) {{
                            child.morphTargetInfluences[0] = visemes[phoneme] || 0;
                        }}
                    }});
                }}
                
                function animate() {{
                    requestAnimationFrame(animate);
                    renderer.render(scene, camera);
                }}
                animate();
            </script>
        </body>
        </html>
        """
        self.viewer.setHtml(html)

    def sync_with_audio(self, audio_data: np.ndarray):
        """Синхронизация анимации губ с аудио"""
        if not self.use_3d_avatar:
            logging.warning("3D avatar is disabled. Skipping animation.")
            return

        if self.animation_thread and self.animation_thread.is_alive():
            return

        def analyze_and_animate():
            # Упрощенный анализ аудио (FFT)
            fft_result = fft(audio_data)
            dominant_freq = np.argmax(np.abs(fft_result))
            # Маппинг частот на виземы
            phoneme = 'sil'
            if dominant_freq < 500:
                phoneme = 'aa'
            elif 500 <= dominant_freq < 1000:
                phoneme = 'E'
            else:
                phoneme = 'SS'
            self._update_animation(phoneme)

        self.animation_thread = threading.Thread(target=analyze_and_animate)
        self.animation_thread.start()

    def _update_animation(self, phoneme: str):
        """Обновление анимации через JS"""
        if not self.use_3d_avatar:
            return

        js_code = f"animateViseme('{phoneme}');"
        self.viewer.page().runJavaScript(js_code)

    def set_emotion(self, emotion: str):
        """Смена эмоции аватара"""
        if not self.use_3d_avatar:
            logging.warning("3D avatar is disabled. Emotion change skipped.")
            return

        self.config.emotion = emotion
        logging.info(f"Emotion set to: {emotion}")

    def save_state(self, path: str):
        """Сохранение текущего состояния аватара"""
        if not self.use_3d_avatar:
            logging.warning("3D avatar is disabled. State saving skipped.")
            return

        state = {
            'camera_position': self.viewer.page().runJavaScript("camera.position.toArray()"),
            'current_animation': self.config.emotion
        }
        with open(path, 'w') as f:
            json.dump(state, f)


# Пример использования
if __name__ == "__main__":
    config = AvatarConfig(
        model_url="https://models.readyplayer.me/your-avatar.glb",
        viseme_map={
            'sil': 0.0, 'PP': 0.3, 'FF': 0.2,
            'TH': 0.4, 'DD': 0.35, 'kk': 0.5
        }
    )

    avatar = AvatarController(config)

    # Тестовый цикл для проверки работы аватара
    print("Введите текст для воспроизведения анимации аватара. Для выхода введите 'стоп'.")
    while True:
        user_input = input("Введите текст: ").strip().lower()
        if user_input in ["стоп", "stop"]:
            print("Завершение работы...")
            break

        # Симуляция аудио данных
        audio_data = np.random.randn(44100)
        avatar.sync_with_audio(audio_data)

    QApplication.instance().exec_()