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
  Open terminal in your root folder and then type " python -m venv venv"
  Then " source venv/bin/activate " for mac/linux or "venv\Scripts\activate "  for windows

2.Install all the requirements
  Type  "pip install -r requirements.txt"

3.Run the app
  Type  "python main.py"



