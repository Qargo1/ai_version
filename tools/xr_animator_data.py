import socket
import asyncio
import websockets
import subprocess

async def handle_connection(websocket, path):
    # Подключаемся к XR Animator
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 39539))
    while True:
        data, addr = sock.recvfrom(65535)  # Получаем данные
        await websocket.send(data.decode())  # Отправляем данные в браузер

# Запускаем XR Animator
def start_xr_animator():
    try:
        # Укажите путь к исполняемому файлу XR Animator
        xr_animator_path = "tools/xr_animator/XR Animator - electron-v32.0.1-win32-x64_SA/electron.exe"  # Или другой формат файла
        subprocess.Popen([xr_animator_path], shell=True)
        print("XR Animator запущен.")
    except Exception as e:
        print(f"Не удалось запустить XR Animator: {e}")

# Основной код
if __name__ == "__main__":
    # Запускаем XR Animator
    start_xr_animator()

    # Запускаем WebSocket-сервер
    async def main():
        start_server = websockets.serve(handle_connection, "127.0.0.1", 8765)
        print("WebSocket-сервер запущен на ws://127.0.0.1:8765")
        await start_server
        await asyncio.Future()  # Бесконечный цикл для поддержания работы сервера

    asyncio.run(main())  # Запускаем событийный цикл <button class="citation-flag" data-index="9">