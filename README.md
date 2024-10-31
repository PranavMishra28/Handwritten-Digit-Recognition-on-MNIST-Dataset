# Handwritten Digit Recognition on MNIST Dataset

This project uses a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset. It provides a simple web interface where users can draw digits, and the model will predict the drawn digit.

## Project Structure
```
Handwritten-Digit-Recognition/
│
├── Workflow.png        # Visual representation of the model workflow
├── app.py              # Flask or pygame-based app to interact with the model
├── bestmmodel.h5       # Trained model saved in HDF5 format
└── code.ipynb          # Jupyter notebook containing code for training the model
```

## Features
- **Model Training:** Uses the MNIST dataset to train a CNN for digit recognition.
- **Drawing Interface:** Allows users to draw digits on the screen and get predictions.
- **Interactive UI:** Predicts digits in real-time using the trained model.
- **Pre-trained Model:** Best performing model saved as `bestmmodel.h5` for inference.

## Technologies Used
- **Python** for programming.
- **Pygame** for creating the drawing interface.
- **Keras/TensorFlow** for building and training the CNN model.
- **Flask (if used)** for the web-based interface.

## Getting Started

### Prerequisites
Make sure you have the following packages installed:
```bash
pip install pygame tensorflow keras numpy opencv-python matplotlib
```

### Running the Drawing Interface (with Pygame)
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Handwritten-Digit-Recognition
   ```

2. Run the Pygame interface:
   ```bash
   python app.py
   ```

3. Draw a digit on the screen and release the mouse button to see the prediction.

### Running the Jupyter Notebook
1. Open `code.ipynb` in Jupyter Notebook.
2. Run the cells to load the dataset, preprocess it, and train the model.
3. Save the best model as `bestmmodel.h5` for future predictions.

## Model Training

- **Dataset:** MNIST dataset with 70,000 grayscale images of handwritten digits (0-9).
- **Architecture:**
  - 2 Convolutional Layers with MaxPooling
  - Dropout for regularization
  - Dense layer with Softmax for output
- **Optimizer:** Adam  
- **Loss Function:** Categorical Crossentropy  
- **Metrics:** Accuracy

## Usage

1. **Drawing Interface:** Draw digits on the canvas using the Pygame app.
2. **Prediction:** The model will resize the drawing to 28x28 pixels, preprocess it, and make a prediction.
3. **Reset:** Press 'n' to clear the canvas and start drawing a new digit.

### Example Code Snippet (app.py)
```python
import pygame, sys, cv2, numpy as np
from keras.models import load_model

MODEL = load_model('bestmmodel.h5')
LABELS = {0: 'Zero', 1: 'One', 2: 'Two', 3: 'Three', 4: 'Four',
          5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine'}

pygame.init()
DISPLAYSURF = pygame.display.set_mode((640, 480))
pygame.display.set_caption('Digit Board')

iswriting = False
number_xcord, number_ycord = [], []

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEMOTION and iswriting:
            x, y = event.pos
            pygame.draw.circle(DISPLAYSURF, (255, 255, 255), (x, y), 4)
            number_xcord.append(x)
            number_ycord.append(y)

        if event.type == pygame.MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == pygame.MOUSEBUTTONUP:
            iswriting = False
            rect = pygame.Rect(min(number_xcord), min(number_ycord),
                               max(number_xcord) - min(number_xcord),
                               max(number_ycord) - min(number_ycord))
            sub = DISPLAYSURF.subsurface(rect)
            img_array = pygame.surfarray.array3d(sub)
            img_array = cv2.resize(img_array, (28, 28))
            img_array = np.mean(img_array, axis=2) / 255.0
            img_array = np.expand_dims(img_array, axis=(0, -1))

            prediction = MODEL.predict(img_array).argmax()
            print(f"Prediction: {LABELS[prediction]}")
            number_xcord, number_ycord = [], []

    pygame.display.update()
```

## License
This project is licensed under the **MIT License**.

## Acknowledgments
- **MNIST Dataset**: Yann LeCun, Corinna Cortes, and Christopher J.C. Burges.
- **Keras/TensorFlow**: Used for building and training the CNN model.

---

This README covers everything from setting up the project to running the Pygame interface or Jupyter Notebook for digit recognition.
