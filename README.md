# CodeAlpha_Handwritten_Character_Recognition-


# Console Version (MNIST Classifier)

## Key Features:
- CNN architecture for digit classification
- MNIST dataset loading and preprocessing
- Model training with 3-epoch optimization
- Test accuracy evaluation
- Random digit prediction with visual confirmation
- Confidence level display

## How It Works:
1. Loads and preprocesses MNIST dataset (60k train, 10k test images)
2. Builds convolutional neural network:
   - 8-filter Conv2D layer with ReLU activation
   - MaxPooling for dimensionality reduction
   - Fully connected output layer with softmax
3. Trains model using Adam optimizer
4. Evaluates on test set and reports accuracy
5. Selects random test image for prediction:
   - Displays digit visualization
   - Compares prediction vs actual label
   - Shows prediction confidence

## Code Structure:
- TensorFlow/Keras for model building
- Matplotlib for visualization
- Sequential model architecture:
  - Conv2D -> MaxPooling2D -> Flatten -> Dense
- Core components:
  - Image normalization (0-255 to 0-1)
  - Sparse categorical crossentropy loss
  - Batch processing for efficient training
- Key variables:
  - 'test_acc': Model accuracy on unseen data
  - 'prediction': Class probability distribution
- Workflow:
  1. Data preparation
  2. Model definition and compilation
  3. Training and evaluation
  4. Single-image prediction demo


# GUI Version (Interactive Recognizer)
## Key Features:
- Interactive canvas for digit drawing
- Real-time digit recognition
- Tkinter GUI interface
- Confidence percentage display
- Canvas clearing functionality
- Pre-trained CNN model integration

## How It Works:
1. Loads and trains same CNN architecture as console version
2. Creates drawing application with:
   - Black canvas (280x280 pixels)
   - "Recognize Digit" prediction button
   - "Clear Canvas" reset button
3. Converts drawings to MNIST-compatible format:
   - Resizes drawings to 28x28 pixels
   - Normalizes pixel values (0-1 range)
4. Runs model inference on user drawings
5. Displays prediction in messagebox:
   - Top predicted digit
   - Confidence percentage

## Code Structure:
- Tkinter for GUI components
- PIL for image processing
- TensorFlow for model inference
- Class-based implementation (DigitRecognizer)
- Event-driven architecture:
  - Mouse drag: Drawing on canvas
  - Buttons: Prediction and clearing
- Core components:
  - Image downsampling (280px → 28px)
  - Drawing to pixel conversion
  - Async prediction (no GUI freeze)
- Key features:
  - Brush size customization (15px radius)
  - Persistent drawing state
  - Model pre-trained offline (3 epochs)
- Workflow:
  1. Model training (background)
  2. GUI initialization
  3. User drawing capture
  4. Image preprocessing
  5. Prediction and result display
