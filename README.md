\# 🩹 Injury Detection System (ML + Streamlit)



This project implements a \*\*binary injury detection system\*\* using deep learning.  

It classifies images of body parts into \*\*Normal\*\* or \*\*Injury\*\* and provides \*\*visual explanations (Grad-CAM)\*\* to help users understand the model’s predictions.



The system is designed as a \*\*screening and decision-support tool\*\*, not a medical diagnosis system.



---



\## 🔍 Features



\- Binary image classification: \*\*Normal vs Injury\*\*

\- Transfer learning using \*\*MobileNetV2\*\*

\- Handling of class imbalance using dataset expansion and class-weighted loss

\- \*\*Class-specific data augmentation\*\* for better generalization

\- Model evaluation using accuracy, precision, recall, and F1-score

\- \*\*Grad-CAM visual explanations\*\* for interpretability

\- Interactive \*\*Streamlit web application\*\* for deployment



---



\## 🧠 Model Overview



\- \*\*Architecture:\*\* MobileNetV2 (pretrained on ImageNet)

\- \*\*Framework:\*\* PyTorch

\- \*\*Training Platform:\*\* Kaggle

\- \*\*Input Size:\*\* 224 × 224 RGB images

\- \*\*Output:\*\* Binary classification (Normal / Injury)

\- \*\*Explainability:\*\* Grad-CAM heatmap overlay



---



\## 📊 Performance Summary



\- Overall test accuracy: ~95%

\- Balanced precision and recall for both classes

\- Reduced false positives after dataset diversification

\- Verified on real-world images using the deployed app



---



\## 🗂 Project Structure

InjuryDetection/

│

├── app/

│ ├── app.py

│ ├── requirements.txt

│ └── model/

│ └── injury\_model.pth

│

├── notebook/

│ └── injury-model(Final).ipynb

│

├── README.md

└── .gitignore


- `app/`: Streamlit deployment code and trained model
- `notebook/`: Model training, experimentation, and evaluation



---



\## 🚀 How to Run the Application



\### 1️⃣ Clone the repository



```bash

git clone https://github.com/ShrenujKarath/InjuryDetection_app.git
cd InjuryDetection_app/app






2️⃣ Install dependencies



pip install -r requirements.txt





3️⃣ Run the Streamlit app



python -m streamlit run app.py



The application will open in your browser at:



http://localhost:8501





🧪 Using the Application



Upload an image of a body part



The app will display:



* Prediction (Normal / Injury)
* Model confidence
* Original image
* Grad-CAM heatmap highlighting influential regions



Use the visualization to understand why the model made the prediction







⚠️ Disclaimer



This system is intended for educational and screening purposes only.

It does not replace professional medical diagnosis or clinical evaluation.



Always consult a qualified medical professional for medical concerns.





📈 Future Improvements



Multi-class injury classification



Larger and more diverse datasets



Confidence score calibration



Mobile deployment



Clinical validation with expert feedback





👤 Author



Shrenuj Karath





