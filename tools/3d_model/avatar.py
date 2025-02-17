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
import sys
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
import subprocess
import urllib.error
import json
import threading
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from urllib.request import urlretrieve
from PyQt5.QtCore import QUrl, QObject, pyqtSignal, QRunnable, QThreadPool
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings
from scipy.fft import fft  # Используем библиотеку scipy для FFT
import logging  # Логирование
#pip install pythreejs
#pip install trimesh

from INFO.previous.cors_server import CORSRequestHandler

import webbrowser
from functools import partial


# Константы
MODEL_NAME = "alleyana.glb"
OS_PATH = os.path.dirname(__file__)
print(f"MODEL_NAME: {MODEL_NAME}, OS_PATH: {OS_PATH}")

THREE_VERSION = "0.150.1"  # Актуальная версия Three.js

class AvatarConfig:
    def __init__(self, path_to_models_url=None, viseme_map=None):
        self.path_to_models_url = path_to_models_url
        self.viseme_map = viseme_map or {}
        

class ModelLoader(QRunnable):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def run(self):
        self.callback()

class AvatarController(QObject):
    animation_updated = pyqtSignal(str)

    def __init__(self, config: AvatarConfig):
        super().__init__()
        self.config = config
        self.local_name = MODEL_NAME
        self.use_3d_avatar = True
        self._init_emotion_map()
        self.viewer = None
        self.server = None
        self.server_thread = None
        self.thread_pool = QThreadPool()

        self._start_local_server()
        self._load_assets()
        self._init_web_view()

        self.animation_updated.connect(self._handle_animation_update)

    def _start_local_server(self):
        """Запуск сервера в отдельном потоке"""
        self.server = HTTPServer(("localhost", 8080), CORSRequestHandler)
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
        logging.info("Server started on http://localhost:8080")

    def _run_server(self):
        try:
            self.server.serve_forever()
        except Exception as e:
            logging.error(f"Server error: {e}")
        finally:
            self.server.server_close()

    def _stop_local_server(self):
        """Корректная остановка сервера"""
        if self.server:
            self.server.shutdown()
            self.server_thread.join(timeout=5)
            logging.info("Server stopped")

    def _load_assets(self):
        """Асинхронная загрузка модели с кэшированием"""
        cache_dir = os.path.join(os.getcwd(), "cached_models")  # Сохраняем в текущей директории
        os.makedirs(cache_dir, exist_ok=True)
        cached_path = os.path.join(cache_dir, os.path.basename(self.local_name))

        if self.config.path_to_models_url:
            if not os.path.exists(cached_path):
                loader = ModelLoader(self._download_model)
                self.thread_pool.start(loader)

    def _download_model(self):
        try:
            logging.error(f"Model download started: url: {self.config.path_to_models_url} + path: {self.local_name}")
            urlretrieve(self.config.path_to_models_url, self.local_name)
            logging.info(f"Model cached: {self.local_name}")
            self._update_model_in_viewer()
        except Exception as e:
            logging.error(f"Model download failed: {e}")
            self.use_3d_avatar = False

    def _update_model_in_viewer(self):
        """Обновление модели во вьювере после загрузки"""
        if self.viewer:
            js_code = f"""
                loader.load({self.config.path_to_models_url}{os.path.basename(self.local_name)})', 
                    function(gltf) {{
                        scene.clear();
                        avatar = gltf.scene;
                        scene.add(avatar);
                    }});
            """
            self.viewer.page().runJavaScript(js_code)

    def _init_emotion_map(self):
        """Инициализация карты эмоций"""
        default_map = {
            'sil': 0.0, 'PP': 0.3, 'FF': 0.2, 'TH': 0.4,
            'DD': 0.35, 'kk': 0.5, 'CH': 0.4, 'SS': 0.3,
            'nn': 0.25, 'RR': 0.4, 'aa': 0.6, 'E': 0.5
        }
        self.config.viseme_map = {**default_map, **self.config.viseme_map}

    def _init_web_view(self):
        """Инициализация WebGL вьювера с аппаратным ускорением"""
        self.app = QApplication.instance() or QApplication([])
        self.container = QWidget()
        self.viewer = QWebEngineView()
        
        # Включаем аппаратное ускорение
        self.viewer.settings().setAttribute(QWebEngineSettings.WebGLEnabled, True)
        self.viewer.settings().setAttribute(QWebEngineSettings.Accelerated2dCanvasEnabled, True)
        
        self._load_html_template()
        self.container.show()

    def _load_html_template(self):
        """Загрузка внешнего HTML-шаблона"""
        html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
        try:
            with open(html_path, "r", encoding="utf-8") as file:
                html = file.read()
            print(f"Loading model from: {self.config.path_to_models_url}{os.path.basename(self.local_name)}")
            self.viewer.setHtml(html)
            self.viewer.show()
        except FileNotFoundError:
            logging.error("HTML template not found. Please check the path.")

    def _handle_animation_update(self, phoneme):
        """Обновление анимации через WebAssembly при наличии"""
        if self.use_3d_avatar:
            weight = self.config.viseme_map.get(phoneme, 0.0)
            js_code = f"""
                if (mixer) {{
                    mixer.timeScale = {weight};
                }}
            """
            self.viewer.page().runJavaScript(js_code)

    def cleanup(self):
        """Освобождение ресурсов"""
        self._stop_local_server()
        self.viewer.close()
        logging.info("Application shutdown complete")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = AvatarConfig(
        path_to_models_url="http://localhost:8080/models/visual/glb_models/"  # Замените на реальный URL
    )
    
    app = QApplication(sys.argv)
    controller = AvatarController(config)
    sys.exit(app.exec_())