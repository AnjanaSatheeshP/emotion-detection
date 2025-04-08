---

# Emotion Detection Web App

A full-stack Emotion Detection web application built using Flask, DeepFace, Bootstrap, and AJAX. This app allows users to detect emotions from text, images, and videos with real-time, interactive, and user-friendly analysis.

---

## 🔍 Features

- **User Authentication**: Secure signup & login system.
- **Text Emotion Detection**: Analyze emotional tone from user-entered text.
- **Image Emotion Detection**: Detect emotions from uploaded images using facial analysis.
- **Video Emotion Detection**: Process uploaded videos to detect facial emotions frame-by-frame.
- **Dynamic Results**: Real-time emotion visualizations with interactive bar charts.
- **Dashboard Layout**: Responsive, professional UI with a sidebar for navigation.
- **Multilingual Input (Upcoming)**: Support for multiple languages with translation.

---

## 🛠 Tech Stack

- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5, AJAX  
- **Backend**: Python, Flask  
- **AI Model**: DeepFace (Facial Emotion Recognition)  
- **Others**: TextBlob, Transformers, OpenCV, Chart.js, Google Translate API (planned)

---

## 📁 Project Structure

emotion-detection/
│
├── static/
│   ├── css/
│   └── js/
│
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── signup.html
│   └── text_detection.html
│   ├── image_detection.html
│   └── video_detection.html
│   ├── multi_language.html
│   └── userpage.html
│
├── app.py
├── detections/
│   ├── detection.py
│   ├── image_detection.py
│   ├── video_detection.py
├── auth.py
└── README.md


---

## 💡 Sample Use Cases

- Mental health and sentiment tracking  
- Content moderation  
- User feedback analysis  
- Emotion-aware AI applications  

---

## 🧠 Model Details

- **Text Detection**: NLP-based classification (e.g., Watson NLP or transformer-based models like `roberta-base-go_emotions`).
- **Image & Video Detection**: DeepFace for facial emotion recognition. Only one face per frame is processed for accuracy.

---

## 🚀 Future Enhancements

- Live camera-based emotion detection  
- Voice emotion detection   
- Multi-face detection in group photos/videos  
- Mobile-responsive design improvements  

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## 🙏 Acknowledgements

- [DeepFace](https://github.com/serengil/deepface) by Sefik Serengil  
- [IBM Watson NLP](https://www.ibm.com/watson) / [HuggingFace Transformers](https://huggingface.co/transformers)  
- [Bootstrap Icons](https://icons.getbootstrap.com/) & [Google Fonts](https://fonts.google.com/)

---

## 🤝 Contribution

Contributions are welcome! Feel free to fork the repo and submit a pull request.

---

## 👩‍💻 Author

**Anjana Satheesh P**  
📧 anjanasatheesh5203@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/anjana-satheesh-p-746a98276/)