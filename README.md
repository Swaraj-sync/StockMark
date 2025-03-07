# **📈 Flask Stock Prediction Web App**  

🚀 This project is a **Flask-based stock prediction web application** that uses machine learning to predict whether a stock's price will **rise or fall in the next 5 days** based on technical indicators.  

---

## **📌 Features**
✅ Enter a **stock ticker** (e.g., *AAPL, TSLA, NVDA*).  
✅ Fetches **real-time stock data** from Yahoo Finance.  
✅ Computes **technical indicators** (SMA, EMA, RSI, MACD, OBV, etc.).  
✅ Uses a **pre-trained machine learning model** to make predictions.  
✅ **User-friendly web interface** built with HTML & Flask.  
✅ **Deployable on Render or Heroku** for live usage.  

---

## **🛠️ Installation & Setup**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/Swaraj-sync/StockMark.git
cd StockMark
```

### **2️⃣ Create & Activate a Virtual Environment**  
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

### **3️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **4️⃣ Run the Flask App Locally**  
```bash
python app.py
```
Open **http://127.0.0.1:5000/** in your browser.

---

## **🖥️ Deployment on Render**  

1. Push your code to **GitHub/GitLab**.  
2. Go to [Render](https://dashboard.render.com/) and create a **new Web Service**.  
3. Connect it to your GitHub repository.  
4. Set the **Start Command** to:  
   ```
   gunicorn app:app
   ```
5. Deploy & test your app!

---

## **📂 Project Structure**
```
📁 Flask-Stock-Prediction/
│-- 📁 templates/          # HTML templates for UI
│   │-- index.html
│   │-- layout.html
│
│-- 📁 static/             # CSS & JavaScript (if needed)
│-- app.py                 # Flask web application
│-- model.pkl              # Pre-trained ML model
│-- scaler.pkl             # Pre-trained scaler
│-- selector.pkl           # Feature selector
│-- requirements.txt       # Python dependencies
│-- Procfile               # For deployment on Heroku
│-- README.md              # Project documentation
```

---

## **🛠️ Built With**
- **Flask** – Backend Framework  
- **yFinance** – Fetches Stock Data  
- **scikit-learn** – Machine Learning  
- **Gunicorn** – Deployment Server  
- **Render/Heroku** – Hosting  

---

## **👨‍💻 Contributors**
💡 **Swaraj Patil And Amanraj Mishra** 

Feel free to **fork**, **contribute**, or **suggest improvements**! 🚀  

📧 **Contact:** patilswaraj1111@gmail.com

---
