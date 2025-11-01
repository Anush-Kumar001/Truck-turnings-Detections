# 🚛 Truck Turnings Detection System

## 🧩 Project Overview
The **Truck Turnings Detection System** is an intelligent vision-based setup designed to detect and monitor the turning behavior of large vehicles such as trucks, buses, and trailers. By mounting a camera on the vehicle’s front side, the system captures live visuals and determines whether the truck is turning or moving straight. The result is displayed in the driver’s cabin using clear visual indicators — **green** for safe straight movement and **red** for turning — ensuring safer driving in real-world traffic environments.

---

## 🎯 Objective
- To design a **real-time visual detection system** that identifies when a truck is turning.
- To enhance **driver awareness** and reduce accidents caused by poor turn judgment or blind spots.
- To contribute toward the development of **semi-autonomous and autonomous vehicle systems** for heavy transport.
- To create an **affordable and scalable** system that can be integrated into existing commercial vehicles.

---

## ⚙️ How It Works
1. The system uses a camera module mounted on the vehicle to continuously capture frames of the front view.
2. A trained **computer vision model** processes these frames to detect **turning arcs** and **directional changes**.
3. The algorithm then determines:
   - ✅ **Green Light:** Straight movement detected.
   - 🔴 **Red Light:** Turning detected (alert mode).
4. This detection result is displayed in real time on a screen in the **driver’s cabin** to alert the driver instantly.
5. Optional integration with **IoT modules** can transmit this data for fleet analysis and safety logging.

---

## 🌍 Use to Society
- **Reduces road accidents** by assisting drivers in managing turns safely.
- Helps **pedestrians and smaller vehicles** avoid danger near heavy vehicles.
- Increases **efficiency in logistics** by preventing damage or downtime due to turn-related mishaps.
- Paves the way for **autonomous truck technologies** and smarter transport ecosystems.
- Encourages **sustainable mobility** through data-driven insights and adaptive driving behavior.

---

## 📈 Real-World Relevance & Statistical Insight
- As of 2025, **no L5 (fully autonomous) trucks** are in large-scale commercial operation worldwide.
- The autonomous trucking market is rapidly evolving, with **L3–L4 systems** showing major progress.
- The global **autonomous truck market** is expected to grow at a **25.6% CAGR (2024–2030)**.
- Companies such as **TuSimple**, **Waymo**, and **Inceptio Technology** are leading trials for large-scale deployment.
- **Inceptio** currently operates **600+ L4-enabled trucks** in China’s logistics routes.
- Vision-based systems like this project act as **crucial safety enablers** during the transition to full automation.

---

## 🔮 Future Scope
- 🚦 Integration with **GPS and IMU sensors** for precise turn prediction.
- 🧠 Enhanced **AI model training** with larger datasets for improved accuracy.
- ☁️ Cloud and IoT connectivity for **real-time fleet monitoring**.
- 🚚 Incorporation into **ADAS (Advanced Driver Assistance Systems)**.
- ⚡ Onboard **edge AI processing** for instant detection without internet dependency.
- 📊 Data analytics dashboards for monitoring driver behavior and safety trends.

---

## 🧠 Tech Stack
- **Programming Language:** Python
- **Frameworks/Libraries:** OpenCV, NumPy, Torch (optional)
- **Hardware (optional):** USB/ESP32 camera, Raspberry Pi, or compatible embedded controller
- **Output Interface:** Visual dashboard for drivers
---

## 📚 License
This project is open-source and can be used for educational and research purposes. Proper credit must be given when reusing or modifying the content.
