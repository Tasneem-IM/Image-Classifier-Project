import argparse
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import tensorflow_hub as hub

def process_image(image):
    image = tf.image.resize(image, (224, 224)) 
    image = image / 255.0  #  القيم بين 0 و 1
    return image.numpy()

def predict(image_path, model, top_k=5):
    image = Image.open(image_path)  # فتح الصورة
    image = np.asarray(image)
    processed_image = process_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)  # لازم يكون فيه batch dimension
    predictions = model.predict(processed_image)

    top_k_indices = np.argsort(predictions[0])[-top_k:][::-1]  # ترتيب النتائج تنازلياً
    top_k_probs = predictions[0][top_k_indices]
    top_k_classes = [str(index + 1) for index in top_k_indices]  # +1 عشان بعض الملفات تبدأ الترقيم من 1

    return top_k_probs, top_k_classes

def load_class_names(json_file):
    with open(json_file, 'r') as f:
        class_names = json.load(f)
    return class_names

def main():
    parser = argparse.ArgumentParser(description='Predict the class of a flower image')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('model_path', type=str, help='Path to the saved model')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to category label mapping file')

    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model_path, custom_objects={'KerasLayer': hub.KerasLayer})#لو من Hub لازم نعرّف KerasLayer

    class_names = None
    if args.category_names:
        class_names = load_class_names(args.category_names)

    probs, classes = predict(args.image_path, model, args.top_k)

    print("Results:")
    for i in range(len(classes)):
        name = class_names[classes[i]] if class_names else f"Class {classes[i]}"
        prob = probs[i]
        print(f"{i+1}: {name} - probability: {prob:.4f}")  

if __name__ == '__main__':
    main()
