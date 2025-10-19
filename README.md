# 🌸 Flower Image Classifier

A deep learning project built with PyTorch and transfer learning (AlexNet, VGG16) to classify flower images into multiple categories. The model efficiently predicts the top 3 most probable flower types from a test image.

---

# 🚀 Features

Trained using transfer learning with pre-trained CNN architectures.

Handles data preprocessing, train-validation-test splitting, and hyperparameter tuning.

Includes a user-interactive prediction module to test images manually.

Robust handling of missing or mislabeled classes.

Clean console output with top 3 predictions and probabilities.

---

# 🧠 Model Details

Framework: PyTorch

Architectures Used: AlexNet, VGG16

Optimizer: Adam

Loss Function: CrossEntropyLoss

Dataset: Flowers Split Dataset (custom organized version of the Flowers dataset)

---

# ⚙️ How to Run

-Clone the repository

git clone https://github.com/<your-username>/Flower-Image-Classifier.git
cd Flower-Image-Classifier


-Create and activate a virtual environment

python -m venv venv
venv\Scripts\activate  # for Windows


-Install dependencies

pip install -r requirements.txt


-Run training (optional)

python train.py


-Run prediction

python prediction.py


-You’ll be prompted to enter the flower name (folder name) and image file name.

-The program displays the top 3 predicted classes with probabilities.

---

# 📊 Example Output
Enter flower class folder (e.g., daisy, rose): daisy
Enter image file name (e.g., 1.jpg): 1.jpg
---------------------------------------------------
Predictions:
1. Daisy - 40.3%
2. Rose - 32.0%
3. Dandelion - 27.7%
---------------------------------------------------

---

# 📂 Project Structure
-Flower-Image-Classifier/
-│
-├── flowers_split/              # Dataset
-├── train.py                    # Model training script
-├── prediction.py               # Image prediction script (interactive)
-├── checkpoint.pth              # Saved trained model
-├── cat_to_name.json            # Category-to-class name mapping
-├── requirements.txt            # Dependencies
-└── README.md                   # Project documentation

---

# 💡Future Enhancements

Add GUI for image upload and live predictions.

Extend support to more flower species.

Deploy as a web app using Streamlit or Flask.

---

# 👩‍💻 Author

-Ponnaganti Gayathri
-🌐 GitHub Profile
