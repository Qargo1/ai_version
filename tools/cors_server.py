from http.server import HTTPServer, SimpleHTTPRequestHandler
import logging
import os
import mimetypes

# Add MIME types for GLTF
mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("model/gltf+json", ".gltf")

# Путь к текущему файлу (server.py)
current_file_path = __file__
print("Current file path:", current_file_path)

# Директория, в которой находится server.py
current_dir = os.path.dirname(current_file_path)
print("Current directory:", current_dir)

# Подняться на один уровень вверх
base_dir = os.path.dirname(current_dir)
print("Base_dir:", base_dir)

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def guess_type(self, path):
        # Override to set correct MIME types
        if path.endswith(".html"):
            return "text/html; charset=utf-8"
        if path.endswith(".css"):
            return "text/css; charset=utf-8"
        if path.endswith(".js"):
            return "application/javascript; charset=utf-8"
        if path.endswith(".mp3"):
            return "audio/mp3"
        return super().guess_type(path)

    def end_headers(self):
        # Add CORS headers
        allowed_origins = ["http://localhost:8000"]
        origin = self.headers.get("Origin")
        if origin in allowed_origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        
        # Cache control headers
        if self.path.endswith((".css", ".js", ".glb", ".gltf")):
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        else:
            self.send_header("Cache-Control", "no-store, must-revalidate")
        
        super().end_headers()

    def do_OPTIONS(self):
        # Handle preflight requests
        self.send_response(204)  # No content
        allowed_origins = ["http://localhost:8000", "null"]  # Добавляем "null"
        origin = self.headers.get("Origin")
        if origin in allowed_origins or origin is None:
            self.send_header("Access-Control-Allow-Origin", origin or "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def list_directory(self, path):
        # Disable directory listing
        self.send_error(403, "Directory listing not allowed")
        return None

class AvatarServer:
    def __init__(self, host="localhost", port=8000, base_dir="static"):
        self.host = host
        self.port = port
        self.base_dir = base_dir
        self.httpd = None

    def start(self):
        path_to = os.path.abspath(self.base_dir)
        print(f'path_to: {path_to}')
        os.chdir(path_to)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info(f"Serving files from: {self.base_dir}")
        self.httpd = HTTPServer((self.host, self.port), CORSRequestHandler)
        logging.info(f"Server started on http://{self.host}:{self.port}")
        self.httpd.serve_forever()

if __name__ == "__main__":
    PORT = 8000
    BASE_DIR = base_dir
    server = AvatarServer(port=PORT, base_dir=BASE_DIR)
    server.start()