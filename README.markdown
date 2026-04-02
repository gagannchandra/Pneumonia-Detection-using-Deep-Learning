# Pneumonia Detection Using Deep Learning

## Project Overview

This project implements a deep learning model to detect pneumonia from chest X-ray images using a pre-trained VGG16 convolutional neural network (CNN) with transfer learning. The model is trained on a dataset of chest X-ray images to classify them as either "Normal" or "Pneumonia." The project leverages the Keras library with TensorFlow as the backend, utilizing data augmentation and pre-trained weights to achieve high accuracy in detecting pneumonia.

The primary goal is to provide an automated tool that can assist medical professionals in diagnosing pneumonia by analyzing chest X-ray images. The model is designed to be efficient and accurate, making it suitable for deployment in medical diagnostic pipelines.

## Dataset

The dataset used in this project is the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset, which contains chest X-ray images labeled as "Normal" or "Pneumonia." The dataset is organized into two main directories:

- **Training Set** (`chest_xray/train`): Contains 5,232 images, split into two classes:
  - Normal: Images of healthy lungs.
  - Pneumonia: Images of lungs affected by pneumonia.
- **Test Set** (`chest_xray/test`): Contains 624 images, also split into Normal and Pneumonia classes.

The dataset is typically provided as a ZIP file (`Pneumonia Detection Dataset.zip`), which is extracted to the `/content/` directory for processing.

## Project Structure

The project is implemented in a Jupyter Notebook (`Pneumonia_Detection_using_Deep_Learning.ipynb`) and follows these key steps:

1. **Data Preparation**:
   - Extract the dataset from a [ZIP file](https://drive.google.com/file/d/1x1-iwcKyorg93Vg2SbX9Jn5I-8unJI5Q/view?usp=drive_link).
   - Set up paths for training and test directories.
   - Apply data augmentation to the training set to increase model robustness.

2. **Model Architecture**:
   - Use the VGG16 model pre-trained on ImageNet, excluding the top fully connected layers.
   - Freeze the convolutional layers to retain pre-trained weights.
   - Add a Flatten layer and a Dense layer with softmax activation for binary classification (Normal vs. Pneumonia).

3. **Model Training**:
   - Compile the model with categorical cross-entropy loss and the Adam optimizer.
   - Train the model for 5 epochs using the training set, with validation on the test set.
   - Use data augmentation techniques like rescaling, shearing, zooming, and horizontal flipping.

4. **Model Evaluation and Prediction**:
   - Save the trained model as `our_model.h5`.
   - Load the model to make predictions on individual test images.
   - Output predictions indicating whether a person is "safe" (Normal) or "affected with Pneumonia."

## Dependencies

The following Python libraries are required to run the project:

- `tensorflow` (2.18.0)
- `keras` (3.8.0)
- `scipy` (1.14.1)
- `glob2` (0.7)
- `matplotlib`
- `numpy`

These dependencies are installed via the command:
```bash
pip install tensorflow keras scipy glob2
```

## Installation and Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/gagannchandra/Pneumonia-Detection-using-Deep-Learning.git
   cd Pneumonia-Detection-using-Deep-Learning
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.11 or later installed, then run:
   ```bash
   pip install -r requirements.txt
   ```
   Alternatively, install the required packages directly:
   ```bash
   pip install tensorflow keras scipy glob2 matplotlib numpy
   ```
   *Note : if pip install failed beacuse of external or something...*
   USE THIS
   
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
   

4. **Download the Dataset**:
   - Download the Chest X-Ray Images (Pneumonia) dataset from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).
   - Place the ZIP file (`Pneumonia Detection Dataset.zip`) in the project directory or update the `zip_path` variable in the notebook to point to the correct location.

5. **Run the Notebook**:
   - Open the Jupyter Notebook (`Pneumonia_Detection_using_Deep_Learning.ipynb`) in a Jupyter environment (e.g., Google Colab, JupyterLab, or VS Code).
   - Execute the cells sequentially to extract the dataset, train the model, and test predictions.

## Model Details

### Architecture
- **Base Model**: VGG16, pre-trained on ImageNet, with the top layers removed.
- **Input Shape**: 224x224x3 (RGB images resized to 224x224 pixels).
- **Custom Layers**:
  - Flatten layer to convert the 2D feature maps into a 1D vector.
  - Dense layer with 2 units (for binary classification) and softmax activation.
- **Trainable Parameters**: Only the Dense layer is trainable (50,178 parameters), while the VGG16 layers are frozen (14,714,688 non-trainable parameters).

### Training
- **Loss Function**: Categorical cross-entropy.
- **Optimizer**: Adam.
- **Metrics**: Accuracy.
- **Epochs**: 5.
- **Batch Size**: 4.
- **Data Augmentation**:
  - Rescale: Normalize pixel values to [0, 1].
  - Shear Range: 0.2.
  - Zoom Range: 0.2.
  - Horizontal Flip: Enabled.

### Performance
The model achieves the following performance on the test set after 5 epochs:
- **Validation Accuracy**: Up to 93.11% (Epoch 4).
- **Validation Loss**: As low as 0.4947 (Epoch 4).

Note that performance may vary slightly due to the random nature of data augmentation and training.

## Usage

To use the trained model for predictions:

1. **Load the Model**:
   ```python
   from keras.models import load_model
   model = load_model('our_model.h5')
   ```

2. **Prepare an Image**:
   - Load a chest X-ray image and resize it to 224x224 pixels.
   - Convert the image to a NumPy array and preprocess it using VGG16's preprocessing function.

3. **Make a Prediction**:
   ```python
   img = image.load_img('path_to_image.jpeg', target_size=(224, 224))
   imagee = image.img_to_array(img)
   imagee = np.expand_dims(imagee, axis=0)
   img_data = preprocess_input(imagee)
   prediction = model.predict(img_data)
   if prediction[0][0] > prediction[0][1]:
       print('Person is safe.')
   else:
       print('Person is affected with Pneumonia.')
   print(f'Predictions: {prediction}')
   ```

4. **Example Output**:
   For a sample pneumonia image:
   ```
   Person is affected with Pneumonia.
   Predictions: [[0. 1.]]
   ```

## Results

The model successfully classifies chest X-ray images into "Normal" or "Pneumonia" with high accuracy. The use of transfer learning with VGG16 allows the model to leverage pre-trained features, reducing training time and improving performance on a relatively small dataset. The data augmentation techniques help prevent overfitting, making the model more robust to variations in X-ray images.

## Limitations

- **Dataset Size**: While the dataset is substantial, it may not capture all variations in chest X-ray images, potentially limiting generalization to real-world scenarios.
- **Class Imbalance**: The dataset may have an imbalance between Normal and Pneumonia classes, which could affect model performance.
- **Overfitting Risk**: Despite data augmentation, the model shows fluctuations in validation loss (e.g., 1.8476 in Epoch 5), indicating potential overfitting.
- **Hardware Requirements**: Training the model requires significant computational resources, preferably a GPU (e.g., T4 as used in the notebook).

## Future Improvements

- **Increase Epochs**: Train the model for more epochs with early stopping to optimize performance.
- **Address Class Imbalance**: Use techniques like class weighting or oversampling to handle any imbalance in the dataset.
- **Model Fine-Tuning**: Unfreeze some VGG16 layers for fine-tuning to improve feature extraction for chest X-ray images.
- **Alternative Models**: Experiment with other pre-trained models like ResNet or EfficientNet for potentially better performance.
- **Cross-Validation**: Implement k-fold cross-validation to ensure robust evaluation.
- **Real-World Testing**: Validate the model on a larger, more diverse dataset from clinical settings.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a pull request.

Please ensure your code follows the project's coding standards and includes appropriate documentation.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- The [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset providers.
- The Keras and TensorFlow teams for providing robust deep learning libraries.
- The open-source community for valuable resources and tutorials on deep learning and medical image analysis.
