from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import os
import zipfile

app = Flask(__name__)

# Caminho absoluto para o modelo salvo
model_path = os.path.join(os.path.dirname(__file__), 'modelo.h5')

# Carregar o modelo Keras salvo
model = load_model(model_path)

def preprocess_image(image, target_size):
    """Função para pré-processar a imagem para o modelo."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalizar os pixels
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def index():
    """Rota inicial que renderiza uma página HTML."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Rota para fazer previsões com o modelo carregado."""
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        # Criar um diretório temporário para armazenar as imagens extraídas
        temp_dir = 'temp_images'
        os.makedirs(temp_dir, exist_ok=True)

        # Descompactar o arquivo ZIP no diretório temporário
        with zipfile.ZipFile(io.BytesIO(file.read()), 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Iterar sobre as imagens no diretório e fazer previsões
        predictions = {}
        for image_name in os.listdir(temp_dir):
            image_path = os.path.join(temp_dir, image_name)
            img = Image.open(image_path)
            processed_image = preprocess_image(img, target_size=(150, 150))

            # Fazer a previsão
            raw_prediction = model.predict(processed_image)
            prediction_value = raw_prediction[0][0]  # Assumindo que o modelo retorna uma previsão binária

            # Lidar com o valor de previsão corretamente
            threshold = 0.5
            prediction = 1 if prediction_value >= threshold else 0

            # Armazenar a previsão com o nome da imagem
            predictions[image_name] = prediction

        # Limpar o diretório temporário
        for image_name in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, image_name))
        os.rmdir(temp_dir)

        # Retornar as previsões como resposta JSON
        return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)

