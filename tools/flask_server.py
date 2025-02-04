from flask import Flask, request, jsonify
from main import ChatBot  # Импортируем твой класс ChatBot

app = Flask(__name__)

# Инициализация чат-бота
chatbot = ChatBot()

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        messages = data.get('messages')  # Получаем массив сообщений
        if not messages or not isinstance(messages, list):
            return jsonify({"error": "Invalid input: 'messages' is required and must be a list."}), 400

        # Генерация ответа
        context = "\n".join([msg['content'] for msg in messages])
        response = chatbot.generate_response(context)

        return jsonify({"message": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)