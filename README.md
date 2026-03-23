

# Airport Queue Management System (Edge AI Multi-Stream)

## Overview
This project is an Edge AI–based Airport Queue Management System designed to monitor passenger flow in real time using multiple camera streams.

It uses a hybrid hardware pipeline:
- NPU for AI inference
- iGPU for video decoding
- CPU for tracking, analytics, and API services

The system detects people, tracks them across frames, and computes queue-related metrics such as waiting time and throughput.

---

## Features

### Core Features
- Multi-camera support (RTSP and video files)
- Real-time person detection using YOLOv8 (OpenVINO INT8)
- Multi-object tracking using ByteTrack
- Queue analytics with zone-based logic
- Interactive dashboard with live video streams

### Zone-Based Tracking
Users can define zones directly from the dashboard:
- Queue Zone (waiting passengers)
- Service Zone (active service area)
- Exit Zone (passengers leaving)

---

### Metrics
Per camera stream:
- Queue Length
- Average Waiting Time
- Average Processing Time
- Throughput per hour


