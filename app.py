from flask import Flask, request, render_template, url_for, send_file
import tensorflow as tf
from tensorflow.keras.applications import VGG19
import os

app = Flask(__name__)

# Load VGG19 model once when the app starts
vgg = VGG19(include_top=False, weights='imagenet')
vgg.trainable = False

# Define the layers to extract features from
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1']
outputs = [vgg.get_layer(layer).output for layer in (content_layers + style_layers)]
model = tf.keras.Model(inputs=vgg.input, outputs=outputs)

def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (128, 128))  # Reduced resolution
    image = image[tf.newaxis, :]  # Add batch dimension
    return image

def content_loss(content, generated):
    return tf.reduce_mean(tf.square(content - generated))

def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True) / tf.cast(n, tf.float32)
    return gram

def style_loss(style, generated):
    style_loss = 0
    for s, g in zip(style, generated):
        style_gram = gram_matrix(s)
        generated_gram = gram_matrix(g)
        style_loss += tf.reduce_mean(tf.square(style_gram - generated_gram))
    return style_loss

def get_feature_representations(content_image, style_image, generated_image):
    # Stack images and pass through the model
    stack = tf.concat([content_image, style_image, generated_image], axis=0)
    outputs = model(stack)
    # Extract content features for content_image
    content_features = outputs[0][0:1]  # First image in stack
    # Extract style features for style_image
    style_features = [output[1:2] for output in outputs[1:3]]  # Second image in stack
    # Extract generated content features for generated_image
    generated_content_features = outputs[0][2:3]  # Third image in stack
    # Extract generated style features for generated_image
    generated_style_features = [output[2:3] for output in outputs[1:3]]  # Third image in stack
    return content_features, style_features, generated_content_features, generated_style_features

def save_image(image, filename):
    image = tf.squeeze(image, axis=0)  # Remove batch dimension
    image = tf.image.convert_image_dtype(image, tf.uint8)
    image = tf.image.encode_png(image)
    tf.io.write_file(filename, image.numpy())

def run_style_transfer(content_image, style_image):
    generated_image = tf.Variable(content_image)
    optimizer = tf.optimizers.Adam(learning_rate=0.01)
    for epoch in range(10):  # Reduced number of epochs
        with tf.GradientTape() as tape:
            content_features, style_features, generated_content_features, generated_style_features = get_feature_representations(
                content_image, style_image, generated_image
            )
            c_loss = content_loss(content_features, generated_content_features)
            s_loss = style_loss(style_features, generated_style_features)
            total_loss = 1e-4 * c_loss + 1e-2 * s_loss
        gradients = tape.gradient(total_loss, generated_image)
        optimizer.apply_gradients([(gradients, generated_image)])
    output_image = (generated_image + 1.0) / 2.0  # Denormalize
    return output_image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            content_image_file = request.files['content']
            style_image_file = request.files['style']
            content_path = 'uploads/content.jpg'
            style_path = 'uploads/style.jpg'
            content_image_file.save(content_path)
            style_image_file.save(style_path)
            content_image = load_image(content_path)
            style_image = load_image(style_path)
            output_image = run_style_transfer(content_image, style_image)
            output_path = 'static/output.png'
            save_image(output_image, output_path)
            os.remove(content_path)
            os.remove(style_path)
            return url_for('static', filename='output.png')
        except Exception as e:
            print(f'An error occurred: {e}')
            return 'An error occurred during style transfer.', 500
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)