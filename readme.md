# 🚦 Smart Traffic Monitoring System

A real-time, AI-powered Smart Traffic Dashboard built with **Flask**, **YOLOv8**, **Socket.IO**, and **SQLite**, capable of detecting vehicles (cars, buses, trucks, motorcycles) from video feeds and visualizing traffic trends live.

---

## 🌟 Features

- ✅ Vehicle detection using YOLOv8 (`ultralytics`)
- ✅ Real-time dashboard with Flask + Socket.IO       #Used Socket.IO to simulate MQTT messaging 
- ✅ Tracks vehicle type counts (car, bus, truck, motorcycle)
- ✅ Traffic trend chart (Chart.js)
- ✅ Annotated detection video preview
- ✅ SQLite-powered data logging with SQLAlchemy
- ✅ Modular, production-ready folder structure

---

## Steps to run 
1. Create a virtual environment and activate
    python -m venv venv
    source venv/bin/activate       # Linux/macOS
    venv\Scripts\activate          # Windows

2.Install all the requirements
    pip install -r requirements.txt

3.Run the app
    python main.py



