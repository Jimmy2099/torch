import onnxruntime as ort
import numpy as np
import os
import csv
import glob
import tempfile
import subprocess
import random

class AutoEncoderONNX:
    def __init__(self, model_path, data_dir="py/data"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        self.session = ort.InferenceSession(model_path)

        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_name = self.session.get_outputs()[0].name

        print(f"Model loaded successfully. Input names: {self.input_names}")
        print(f"Output name: {self.output_name}")

        self.parameters = self.load_parameters(data_dir)

        for param_name, param_data in self.parameters.items():
            print(f"Parameter {param_name}: shape {param_data.shape}")

    def load_parameters(self, data_dir):
        parameters = {}

        layer_specs = [
            ("encoder.0", [128, 784], [128]),
            ("encoder.2", [64, 128], [64]),
            ("encoder.4", [32, 64], [32]),
            ("decoder.0", [64, 32], [64]),
            ("decoder.2", [128, 64], [128]),
            ("decoder.4", [784, 128], [784]),
        ]

        for name, weight_shape, bias_shape in layer_specs:
            weight_path = os.path.join(data_dir, f"{name}.weight.csv")
            if os.path.exists(weight_path):
                weight_data = self.load_tensor_from_csv(weight_path, weight_shape)
                weight_data = weight_data.reshape(weight_shape).T
                parameters[f"{name}.weight"] = weight_data.astype(np.float32)
                print(f"Loaded weight {name}: {weight_data.shape}")
            else:
                print(f"Warning: Weight file not found: {weight_path}")

            bias_path = os.path.join(data_dir, f"{name}.bias.csv")
            if os.path.exists(bias_path):
                bias_data = self.load_tensor_from_csv(bias_path, bias_shape)
                parameters[f"{name}.bias"] = bias_data.astype(np.float32)
                print(f"Loaded bias {name}: {bias_data.shape}")
            else:
                print(f"Warning: Bias file not found: {bias_path}")

        return parameters

    def load_tensor_from_csv(self, path, shape):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                data.extend([float(x) for x in row])

        return np.array(data).reshape(shape)

    def preprocess(self, image_data):
        data = np.array(image_data, dtype=np.float32)

        if data.size == 784:
            data = data.reshape(1, 784)
        else:
            raise ValueError("Input data size must be 784 (28x28)")
        return data

    def predict(self, input_data):
        processed_data = self.preprocess(input_data)
        print(f"Input data shape: {processed_data.shape}")

        input_feed = {"input": processed_data}

        for param_name, param_data in self.parameters.items():
            input_feed[param_name] = param_data

        try:
            outputs = self.session.run(
                [self.output_name],
                input_feed
            )
            print(f"Output shape: {outputs[0].shape}")
            return outputs[0]
        except Exception as e:
            print(f"Error during inference: {e}")
            raise

    def save_to_csv(self, data, file_path):
        reshaped = data.reshape(28, 28)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["shape"] + list(reshaped.shape))
            for row in reshaped:
                writer.writerow(row)
        print(f"Saved denoised image to {file_path}")

def load_image_from_csv(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)

        first_row = next(reader, None)
        if first_row and first_row[0] == "shape":
            for row in reader:
                data.extend([float(x) for x in row])
        else:
            if first_row:
                data.extend([float(x) for x in first_row])
            for row in reader:
                data.extend([float(x) for x in row])

    if len(data) != 784:
        raise ValueError(f"Expected 784 values in CSV, got {len(data)}")

    return np.array(data)

def load_data_from_csv_dir(directory):
    images = []
    labels = []

    label_file = os.path.join(directory, "labels.csv")
    label_map = {}

    if os.path.exists(label_file):
        with open(label_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    filename = row[0].strip()
                    label = row[1].strip()
                    label_map[filename] = label

    csv_files = glob.glob(os.path.join(directory, "*.png.csv"))
    if not csv_files:
        csv_files = glob.glob(os.path.join(directory, "*.csv"))
        csv_files = [f for f in csv_files if not f.endswith("labels.csv")]

    for file_path in csv_files:
        try:
            image_data = load_image_from_csv(file_path)
            filename = os.path.basename(file_path)

            images.append(image_data)
            labels.append(filename)
            print(f"Successfully loaded {filename}")
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")

    return images, labels

def predict_plot(image_paths):
    suffix = str(random.randint(0, 1000000))
    tmp_script = os.path.join(tempfile.gettempdir(), f"predict_plot_{suffix}.py")
    tmp_data = os.path.join(tempfile.gettempdir(), f"image_predictions_{suffix}.txt")

    python_script = '''
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np

def load_numpy_from_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip()
        if not header.startswith("shape,"):
            # Fallback for old format
            f.seek(0)
            data = np.loadtxt(f, delimiter=",")
            return data

        shape_str = header.split(",")[1:]
        if not all(s.isdigit() for s in shape_str):
             raise ValueError("Invalid shape format in CSV header")

        shape = tuple(map(int, shape_str))
        data = np.loadtxt(f, delimiter=",")
    
    return data.reshape(shape)

def load_data(file_path):
    image_paths = []
    predictions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 2:
                image_paths.append(parts[0])
                predictions.append(parts[1])
    return image_paths, predictions

def predict_plot(data_file):
    image_paths, predictions = load_data(data_file)
    num = len(image_paths)
    if num == 0:
        print("No valid image-prediction pairs found to plot.")
        return
        
    fig, axes = plt.subplots(2, num, figsize=(15, 5)) 

    # Handle case where there is only one image
    if num == 1:
        axes = np.array(axes).reshape(2, 1)

    for i, (img_path, denoise_csv_path) in enumerate(zip(image_paths, predictions)):
        if not os.path.exists(img_path):
            print(f"Warning: Image file not found: {img_path}")
            axes[0,i].text(0.5, 0.5, 'Original not found', ha='center', va='center')
            axes[0,i].axis('off')
        else:
            try:
                img = mpimg.imread(img_path)
                axes[0,i].imshow(img, cmap='gray')
                axes[0,i].set_title("Original")
            except Exception as e:
                print(f"Error reading image {img_path}: {e}")
                axes[0,i].text(0.5, 0.5, 'Error loading image', ha='center', va='center')

        if not os.path.exists(denoise_csv_path):
            print(f"Warning: Denoised CSV file not found: {denoise_csv_path}")
            axes[1,i].text(0.5, 0.5, 'Denoised not found', ha='center', va='center')
            axes[1,i].axis('off')
        else:
            try:
                image_data = load_numpy_from_csv(denoise_csv_path)
                axes[1,i].imshow(image_data, cmap='gray')
                axes[1,i].set_title("Denoised")
            except Exception as e:
                print(f"Error loading denoised CSV {denoise_csv_path}: {e}")
                axes[1,i].text(0.5, 0.5, 'Error loading CSV', ha='center', va='center')

        axes[0,i].axis('off')
        axes[1,i].axis('off')

    plt.tight_layout()
    plt.savefig("predictions.png")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_plot.py <data_file_path>")
        sys.exit(1)
    data_file = sys.argv[1]
    predict_plot(data_file)
'''

    with open(tmp_script, 'w', encoding='utf-8') as f:
        f.write(python_script)

    with open(tmp_data, 'w', encoding='utf-8') as f:
        for img_path in image_paths:
            if not os.path.exists(img_path):
                print(f"Warning: Original image not found: {img_path}")
                continue

            denoise_filename = os.path.basename(img_path).replace(".png", ".png.denoise.csv")
            denoise_path = os.path.join("results", denoise_filename)

            if not os.path.exists(denoise_path):
                print(f"Warning: Denoised file not found: {denoise_path}")
                continue

            f.write(f"{img_path},{denoise_path}\n")

    try:
        result = subprocess.run(['python', tmp_script, tmp_data],
                                capture_output=True, text=True, check=False)
        print("Python script output:")
        print(result.stdout)
        if result.stderr:
            print("Python script errors:")
            print(result.stderr)

        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        if os.path.exists("predictions.png"):
            # 解决windows下rename时目标文件已存在的问题
            target_path = os.path.join(results_dir, "predictions.png")
            if os.path.exists(target_path):
                os.remove(target_path)
            os.rename("predictions.png", target_path)
            print(f"Plot saved to {target_path}")

    finally:
        if os.path.exists(tmp_script):
            os.remove(tmp_script)
        if os.path.exists(tmp_data):
            os.remove(tmp_data)

if __name__ == "__main__":
    MODEL_PATH = "ae_model.onnx"
    DATA_DIR = "py/mnist_noisy_images"
    PARAM_DIR = "py/data"
    RESULTS_DIR = "results"

    try:
        if not os.path.exists(PARAM_DIR):
            print(f"Parameter directory {PARAM_DIR} not found. Please check the path.")
            exit(1)

        autoencoder = AutoEncoderONNX(MODEL_PATH, PARAM_DIR)

        images, labels = load_data_from_csv_dir(DATA_DIR)
        print(f"Loaded {len(images)} images from {DATA_DIR}")

        if len(images) == 0:
            print("No images found. Please check the data directory.")
            exit(1)

        os.makedirs(RESULTS_DIR, exist_ok=True)

        image_paths = []
        for i, (image_data, label) in enumerate(zip(images, labels)):
            print(f"Processing image {i+1}/{len(images)}: {label}")

            denoised = autoencoder.predict(image_data)

            if label.endswith(".png.csv"):
                output_filename = label.replace(".png.csv", ".png.denoise.csv")
            else:
                output_filename = label.replace(".csv", ".denoise.csv")

            output_path = os.path.join(RESULTS_DIR, output_filename)
            autoencoder.save_to_csv(denoised[0], output_path)

            if label.endswith(".png.csv"):
                original_image_path = os.path.join(DATA_DIR, label.replace(".png.csv", ".png"))
            else:
                original_image_path = os.path.join(DATA_DIR, label.replace(".csv", ".png"))

            if os.path.exists(original_image_path):
                image_paths.append(original_image_path)
            else:
                print(f"Warning: Original image not found: {original_image_path}")

        if image_paths:
            print("Generating plot...")
            predict_plot(image_paths)
            print("Predictions saved and plotted successfully")
        else:
            print("No original images found to generate a plot.")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()