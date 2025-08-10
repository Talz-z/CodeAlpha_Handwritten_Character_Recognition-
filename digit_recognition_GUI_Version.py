import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
from tkinter import messagebox


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=3, verbose=0)


class DigitRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognizer")
        
        
        self.canvas = tk.Canvas(root, width=280, height=280, bg="black")
        self.canvas.pack()
        
       
        self.predict_btn = tk.Button(root, text="Recognize Digit", command=self.predict_digit)
        self.predict_btn.pack(pady=10)
        
       
        self.clear_btn = tk.Button(root, text="Clear Canvas", command=self.clear_canvas)
        self.clear_btn.pack(pady=5)
        
       
        self.image = Image.new("L", (280, 280), "black")
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

    def draw_on_canvas(self, event):
        x, y = event.x, event.y
        r = 15  
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill="white")
    
    def reset(self, event):
        pass
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), "black")
        self.draw = ImageDraw.Draw(self.image)
    
    def predict_digit(self):
       
        small_img = self.image.resize((28, 28))
        
        
        img_array = np.array(small_img).reshape(1, 28, 28, 1).astype('float32') / 255.0
        
      
        prediction = model.predict(img_array, verbose=0)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        
        messagebox.showinfo("Prediction", 
                           f"i think this is a: {digit}\nConfidence: {confidence:.1%}")


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizer(root)
    root.mainloop()