import json
import asyncio
from pythonosc import dispatcher, osc_server
import time

# Глобальные переменные
recorded_messages = []
previous_messages = set()  # Для отслеживания уникальных сообщений
duplicate_count = 0  # Счётчик последовательных дубликатов
LOOP_THRESHOLD = 4  # Порог для завершения записи
MAX_DUPLICATE_COUNT = 65
loop_count = 0

PATH_TO_ANIMATION = "static/animations/stand_relaxed.json"

EMOTION_MAPPING = {
    "a": "aa",  # Гласная "а"
    "i": "ih",  # Гласная "и"
    "u": "ou",  # Гласная "у"
    "e": "ee",  # Гласная "е"
    "o": "oh",  # Гласная "о"
    "blink": "blink",  # Моргание
    "blink_l": "blinkLeft",  # Моргание левым глазом
    "blink_r": "blinkRight",  # Моргание правым глазом
    "fun": "neutral",  # Нейтральная эмоция
    "sorrow": "sad",  # Грусть
    "angry": "angry",  # Злость
    "joy": "happy",  # Радость
    "Surprised": "surprised",  # Удивление
    "lookLeft": "lookLeft",  # Взгляд влево
    "lookRight": "lookRight",  # Взгляд вправо
    "lookUp": "lookUp",  # Взгляд вверх
    "lookDown": "lookDown"  # Взгляд вниз
}

BONE_MAPPING = {
    "LeftThumbIntermediate": "leftThumbProximal",
    "RightThumbIntermediate": "rightThumbProximal",
    # Добавьте остальные соответствия
}

def transform_data(input_file, output_file):
    with open(input_file, "r") as file:
        data = json.load(file)

    bones = {}
    start_time = time.time()  # Начальное время (время старта программы)
    last_time = 0  # Начальное время

    for i, frame in enumerate(data):
        if frame["address"] == "/VMC/Ext/Bone/Pos":
            bone_name = frame["args"][0]
            x, y, z, qx, qy, qz, qw = frame["args"][1:]

            # Когда вы получаете данные для анимации
            current_time = (time.time() - start_time) * 1000  # Время в миллисекундах с момента старта

            # Проверка на изменения времени
            if current_time == last_time:
                continue  # Пропускаем, если время не изменилось
            last_time = current_time

            if bone_name not in bones:
                bones[bone_name] = []
            bones[bone_name].append({
                "time": current_time,
                "position": [x, y, z],
                "rotation": [qx, qy, qz, qw]
            })

    output_data = {"bones": []}
    for bone_name, keyframes in bones.items():
        output_data["bones"].append({
            "name": bone_name,
            "keyframes": keyframes
        })

    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)

    print(f"Animation data saved to {output_file}")

def rename_emotion(emotion_name):
    """
    Преобразует имя эмоции согласно карте соответствия.
    Если имя не найдено, возвращает исходное имя.
    """
    return EMOTION_MAPPING.get(emotion_name, emotion_name)

def rename_bone(emotion_name):
    """
    Преобразует имя кости согласно карте соответствия.
    Если имя не найдено, возвращает исходное имя.
    """
    return BONE_MAPPING.get(emotion_name, emotion_name)

def to_lower_case_first_letter(name):
    if not name:
        return name
    return name[0].lower() + name[1:]

def handle_osc_message(address, *args):
    global duplicate_count
    global loop_count

    # Преобразуем имя кости или эмоции
    if address == "/VMC/Ext/Bone/Pos":
        bone_name = args[0]
        renamed_bone = to_lower_case_first_letter(rename_bone(bone_name))
        transformed_args = (renamed_bone, *args[1:])
    elif address == "/VMC/Ext/Blend/Val":
        emotion_name = args[0]
        renamed_emotion = rename_emotion(emotion_name)
        transformed_args = (renamed_emotion, *args[1:])
    else:
        transformed_args = args

    # Формируем сообщение
    message = {"address": address, "args": transformed_args}

    # Проверяем, является ли сообщение дубликатом
    message_str = json.dumps(message)
    if message_str in previous_messages:
        duplicate_count += 1
    else:
        if duplicate_count > 62:
            #print(f"Message was {message}")
            print(f"Duplicate count was {duplicate_count}")
        if duplicate_count >= MAX_DUPLICATE_COUNT:
            loop_count += 1
        duplicate_count = 0  # Сбрасываем счётчик, если новое сообщение
        previous_messages.add(message_str)
    
    recorded_messages.append(message)

    # Логируем первые несколько сообщений для отладки
    if len(recorded_messages) <= 5 or duplicate_count == 100:
        print(f"Получено сообщение: {message}")

    # Если достигнут порог дубликатов, завершаем запись
    if loop_count >= LOOP_THRESHOLD:
        save_to_file()

# Сохранение данных в файл
def save_to_file(filename=PATH_TO_ANIMATION):
    with open(filename, "w") as file:
        json.dump(recorded_messages, file, indent=4) 
    transform_data(PATH_TO_ANIMATION, PATH_TO_ANIMATION)
    print(f"Анимация сохранена в файл: {filename}")
    exit(0)  # Завершаем программу после сохранения

# Запуск OSC-сервера
async def start_osc_server():
    disp = dispatcher.Dispatcher()
    disp.set_default_handler(handle_osc_message)  # Используем обработчик
    server = osc_server.AsyncIOOSCUDPServer(("127.0.0.1", 39539), disp, asyncio.get_event_loop())
    transport, _ = await server.create_serve_endpoint()
    print(f"OSC-сервер запущен на порту 39539")
    return transport

# Основная функция
async def main():
    transport = await start_osc_server()
    try:
        while True:
            await asyncio.sleep(1)  # Бесконечный цикл
    except KeyboardInterrupt:
        print("Программа завершена пользователем.")
    finally:
        transport.close()
        print("OSC-сервер закрыт")

if __name__ == "__main__":
    asyncio.run(main())  # Запускаем программу