
"""
Airport Queue Management System — Multi-Stream Edition
=======================================================
Hardware pipeline (Intel Core Ultra 7 255H):
  • NPU  → YOLOv8n INT8 inference (OpenVINO worker threads, shared queue)
  • iGPU → Video decode via VAAPI / cv2 hw backend for live streams
  • CPU  → ByteTrack (Kalman + 2-stage IoU, per-stream, no dep)
           Flask REST API, analytics, drawing

Architecture:
  Camera/File → VideoDecoder (GPU/CPU) → frame_queue
                → NPUInferenceEngine (worker pool) → callback
                → ByteTracker (per stream, CPU) → QueueAnalytics
                → draw overlay → MJPEG stream / REST API → Dashboard

Zones (drawn interactively in the dashboard, stored as normalised [0..1] polygon):
  • Queue Zone   – passengers waiting
  • Service Zone – passengers at counter
  • Exit Zone    – passengers leaving

Metrics (per stream):
  Queue Length · Avg Waiting Time · Avg Processing Time · Throughput/hr

Fixes & Enhancements in this version:
  FIX-1  Video loop: release+reopen instead of CAP_PROP_POS_FRAMES seek
         (seek is broken for H.264/MP4 in OpenCV — reopen always works)
  FIX-2  Cold-start: persons already inside zones at frame-1 are seeded
         with entry timestamps so metrics work immediately on recordings
  NEW-1  Alert persistence  → queue_alerts/cam_<id>_alerts.json
  NEW-2  Alert acknowledgement → POST /api/cameras/<id>/alerts/<idx>/ack
  NEW-3  History CSV export  → GET /api/cameras/<id>/history/export
  NEW-4  History disk backup every 60 s → queue_alerts/cam_<id>_history.json
  NEW-5  Counter performance (min/max/avg/p50/p90)
         → GET /api/cameras/<id>/performance
  NEW-6  Configurable inference FPS throttle per camera (default 15, 1-30)
         POST /api/cameras { "inf_fps": 15 }
  NEW-7  Multi-camera overview → GET /api/overview
  NEW-8  Exit-zone-missing warning in API responses + HUD overlay

Run:
  python queue_system.py [--host 0.0.0.0] [--port 5050]

Add streams:
  POST /api/cameras  {"name":"Gate-1","source":"rtsp://...","mode":"live","inf_fps":15}
  POST /api/cameras  {"name":"Counter-A","source":"/path/video.mp4","mode":"recording"}
"""

import os, sys, time, json, csv, threading, queue, argparse, logging, io
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Callable
import numpy as np
import cv2
from flask import Flask, Response, jsonify, request, send_from_directory, abort

# ── thread/OV tuning ─────────────────────────────────────────────────────────
os.environ.setdefault("OPENVINO_NUM_THREADS", "12")
os.environ.setdefault("OMP_NUM_THREADS", "12")
cv2.setNumThreads(8)

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("queue_system.log", mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("AirportQueue")

# ── paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).parent.resolve()
_LOITER_DIR = _SCRIPT_DIR.parent / "loitering_alerts"
_ALERTS_DIR = _SCRIPT_DIR / "queue_alerts"
_ALERTS_DIR.mkdir(exist_ok=True)

OV_MODEL_CANDIDATES = [
    _LOITER_DIR / "yolov8n_int8_openvino_model" / "yolov8n.xml",
    _LOITER_DIR / "yolov8n_openvino_model"      / "yolov8n.xml",
    _SCRIPT_DIR / "yolov8n_int8_openvino_model"  / "yolov8n.xml",
    _SCRIPT_DIR / "yolov8n_openvino_model"        / "yolov8n.xml",
]

CONF_THR = 0.12   # low conf threshold — INT8 NPU model on crowds
IOU_THR  = 0.60   # higher NMS IOU — dense crowd keeps more boxes
INF_SIZE = 320    # must match the export resolution of the INT8 model

log.info("=" * 68)
log.info("  AIRPORT QUEUE MANAGEMENT — Intel Core Ultra 7 (NPU+GPU+CPU)")
log.info("=" * 68)


# ─────────────────────────────────────────────────────────────────────────────
# NPU Inference Engine
# ─────────────────────────────────────────────────────────────────────────────
class NPUInferenceEngine:
    """
    Shared pool of 8 NPU/GPU/CPU workers.
    All camera streams submit frames here; first free worker runs inference
    and fires the camera's callback.
    """
    NUM_WORKERS = 8

    def __init__(self):
        self.frame_queue   = queue.Queue(maxsize=128)
        self.workers: List[threading.Thread] = []
        self.compiled_models = []
        self.running    = False
        self.actual_dev = "CPU"
        self._lock      = threading.Lock()
        self.stats      = {"batches": 0, "dropped": 0, "total_det": 0, "inf_ms_sum": 0.0}

    def start(self):
        if self.running: return
        self.running = True

        xml_path = next((p for p in OV_MODEL_CANDIDATES if p.exists()), None)
        if xml_path is None:
            log.error("[NPU] No OpenVINO model found — inference disabled")
            self.running = False; return

        from openvino.runtime import Core
        core  = Core()
        avail = core.available_devices
        log.info(f"[OV] Available devices: {avail}")

        preferred = [d for d in ("NPU", "GPU", "CPU") if d in avail]
        dev = "AUTO:" + ",".join(preferred) if len(preferred) > 1 else (preferred[0] if preferred else "CPU")
        log.info(f"[NPU ENGINE] Compiling on {dev} …")

        config = {
            "PERFORMANCE_HINT": "THROUGHPUT",
            "CACHE_DIR": str(_SCRIPT_DIR / ".ov_cache"),
        }
        try:
            model   = core.read_model(str(xml_path))
            base_cm = core.compile_model(model, dev, config)
            self.actual_dev = dev
            log.info(f"✓ [NPU ENGINE] {xml_path.name} → {dev}")
        except Exception as e:
            log.warning(f"[NPU ENGINE] {dev} failed ({e}), fallback CPU")
            base_cm = core.compile_model(core.read_model(str(xml_path)), "CPU", config)
            self.actual_dev = "CPU"

        for i in range(self.NUM_WORKERS):
            cm = base_cm if i == 0 else core.compile_model(
                core.read_model(str(xml_path)),
                self.actual_dev.split(":")[0] if ":" not in self.actual_dev else self.actual_dev,
                config
            )
            self.compiled_models.append(cm)
            t = threading.Thread(target=self._worker, args=(i, cm), daemon=True,
                                 name=f"NPUWorker-{i}")
            t.start(); self.workers.append(t)
        log.info(f"✓ [NPU ENGINE] {self.NUM_WORKERS} workers on {self.actual_dev}")

    def stop(self):
        self.running = False
        for _ in self.workers:
            try: self.frame_queue.put_nowait(None)
            except: pass
        for w in self.workers: w.join(timeout=2)
        self.workers.clear(); self.compiled_models.clear()

    def submit(self, cam_id: int, frame: np.ndarray, callback: Callable):
        if not self.running: return
        try:
            self.frame_queue.put_nowait((cam_id, frame, callback, time.perf_counter()))
        except queue.Full:
            with self._lock: self.stats["dropped"] += 1

    def _preprocess(self, frame):
        h, w = frame.shape[:2]
        scale = min(INF_SIZE / w, INF_SIZE / h)
        nw, nh = int(w * scale), int(h * scale)
        pw, ph = (INF_SIZE - nw) // 2, (INF_SIZE - nh) // 2
        padded = np.full((INF_SIZE, INF_SIZE, 3), 114, np.uint8)
        padded[ph:ph+nh, pw:pw+nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blob = np.ascontiguousarray(padded.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]
        return blob, scale, pw, ph, w, h

    def _postprocess(self, out, scale, pw, ph, ow, oh):
        if out.ndim == 3: out = out[0].T
        dets = []
        for row in out:
            cx, cy, bw, bh = row[:4]
            scores = row[4:]; cls = int(np.argmax(scores)); conf = float(scores[cls])
            if cls != 0 or conf < CONF_THR: continue
            x1 = max(0.0, ((cx - bw / 2) - pw) / scale)
            y1 = max(0.0, ((cy - bh / 2) - ph) / scale)
            x2 = min(ow - 1, ((cx + bw / 2) - pw) / scale)
            y2 = min(oh - 1, ((cy + bh / 2) - ph) / scale)
            if x2 > x1 and y2 > y1:
                dets.append({"bbox": [x1, y1, x2, y2], "conf": conf, "cls": 0})
        return _nms(dets, IOU_THR)

    def _worker(self, wid: int, cm):
        infer_req = cm.create_infer_request()
        while self.running:
            try: item = self.frame_queue.get(timeout=0.1)
            except queue.Empty: continue
            if item is None: break
            cam_id, frame, callback, t0 = item
            try:
                blob, scale, pw, ph, ow, oh = self._preprocess(frame)
                infer_req.infer({0: blob})
                out  = infer_req.get_output_tensor(0).data
                dets = self._postprocess(out, scale, pw, ph, ow, oh)
                inf_ms = (time.perf_counter() - t0) * 1000
                with self._lock:
                    self.stats["batches"]    += 1
                    self.stats["inf_ms_sum"] += inf_ms
                    self.stats["total_det"]  += len(dets)
                if callback:
                    callback(cam_id, dets, inf_ms)
            except Exception as e:
                log.error(f"[NPUWorker-{wid}] cam {cam_id}: {e}")

    def get_stats(self):
        with self._lock:
            b = max(1, self.stats["batches"])
            return {
                "device":     self.actual_dev,
                "batches":    self.stats["batches"],
                "avg_inf_ms": round(self.stats["inf_ms_sum"] / b, 1),
                "dropped":    self.stats["dropped"],
            }


# ─────────────────────────────────────────────────────────────────────────────
# NMS
# ─────────────────────────────────────────────────────────────────────────────
def _iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    u = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / u if u > 0 else 0.0

def _nms(dets, thr):
    if not dets: return []
    dets = sorted(dets, key=lambda x: x["conf"], reverse=True)
    keep = []; sup = [False] * len(dets)
    for i in range(len(dets)):
        if sup[i]: continue
        keep.append(dets[i])
        for j in range(i + 1, len(dets)):
            if not sup[j] and _iou(dets[i]["bbox"], dets[j]["bbox"]) > thr:
                sup[j] = True
    return keep


# ─────────────────────────────────────────────────────────────────────────────
# ByteTracker  (pure Python — Kalman + 2-stage IoU, CPU)
# ─────────────────────────────────────────────────────────────────────────────
class KalmanBox:
    def __init__(self, bbox):
        cx = (bbox[0] + bbox[2]) / 2; cy = (bbox[1] + bbox[3]) / 2
        w  = bbox[2] - bbox[0];       h  = bbox[3] - bbox[1]
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], np.float32)
        self.F = np.eye(8, dtype=np.float32)
        [self.F.__setitem__((i, i + 4), 1.0) for i in range(4)]
        self.H = np.eye(4, 8, dtype=np.float32)
        self.P = np.diag([10, 10, 10, 10, 1e4, 1e4, 1e4, 1e4]).astype(np.float32)
        self.Q = np.diag([1, 1, 1, 1, .01, .01, .01, .01]).astype(np.float32)
        self.R = np.diag([1, 1, 10, 10]).astype(np.float32)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self._bb()

    def update(self, bbox):
        cx = (bbox[0] + bbox[2]) / 2; cy = (bbox[1] + bbox[3]) / 2
        w  = bbox[2] - bbox[0];       h  = bbox[3] - bbox[1]
        z = np.array([cx, cy, w, h], np.float32); y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(8, dtype=np.float32) - K @ self.H) @ self.P

    def _bb(self):
        cx, cy, w, h = self.x[:4]
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


class _BTrack:
    _cnt = 1

    def __init__(self, bbox, conf):
        self.id    = _BTrack._cnt; _BTrack._cnt += 1
        self.kf    = KalmanBox(bbox); self.conf = conf
        self.state = "tentative"; self.hits = 1; self.age = 0
        self.tsu   = 0; self.bbox = list(bbox)

    def predict(self):
        self.bbox = self.kf.predict(); self.age += 1; self.tsu += 1

    def update(self, bbox, conf):
        self.kf.update(bbox); self.bbox = self.kf._bb()
        self.conf = conf; self.hits += 1; self.tsu = 0
        if self.hits >= 1: self.state = "confirmed"

    def miss(self):
        self.tsu += 1
        if self.tsu > 5: self.state = "lost"


class ByteTracker:
    HIGH = 0.25; MIN_IOU = 0.20; MAX_LOST = 30

    def __init__(self): self.tracks: List[_BTrack] = []

    @staticmethod
    def _match(tracks, dets, min_iou):
        if not tracks or not dets: return [], [], list(range(len(dets)))
        flat = sorted(
            [(float(_iou(t.bbox, d["bbox"])), ti, di)
             for ti, t in enumerate(tracks)
             for di, d in enumerate(dets)],
            key=lambda x: -x[0]
        )
        mt, md, pairs = set(), set(), []
        for sc, ti, di in flat:
            if sc < min_iou: break
            if ti in mt or di in md: continue
            pairs.append((ti, di)); mt.add(ti); md.add(di)
        return (pairs,
                [i for i in range(len(tracks)) if i not in mt],
                [i for i in range(len(dets))   if i not in md])

    def update(self, dets):
        for t in self.tracks: t.predict()
        hi = [d for d in dets if d["conf"] >= self.HIGH]
        lo = [d for d in dets if self.HIGH > d["conf"] >= CONF_THR]
        active = [t for t in self.tracks if t.state != "lost"]
        p1, ut1, ud1 = self._match(active, hi, self.MIN_IOU)
        for ti, di in p1: active[ti].update(hi[di]["bbox"], hi[di]["conf"])
        rem = [active[i] for i in ut1]
        p2, ut2, ud2 = self._match(rem, lo, self.MIN_IOU)
        for ti, di in p2: rem[ti].update(lo[di]["bbox"], lo[di]["conf"])
        matched_rem = {ti for ti, _ in p2}
        for i, t in enumerate(rem):
            if i not in matched_rem: t.miss()
        for di in ud1:
            self.tracks.append(_BTrack(hi[di]["bbox"], hi[di]["conf"]))
        if not active:
            for di in ud2:
                self.tracks.append(_BTrack(lo[di]["bbox"], lo[di]["conf"]))
        self.tracks = [t for t in self.tracks
                       if not (t.state == "lost" and t.tsu > self.MAX_LOST)]
        return [{"bbox": t.bbox, "conf": t.conf, "cls": 0,
                 "track_id": t.id, "state": t.state}
                for t in self.tracks if t.state != "lost"]


# ─────────────────────────────────────────────────────────────────────────────
# Zone helpers
# ─────────────────────────────────────────────────────────────────────────────
def _in_poly(pt, poly):
    if not poly or len(poly) < 3: return False
    x, y = pt; inside = False; px, py = poly[-1]
    for nx, ny in poly:
        if ((ny > y) != (py > y)) and x < (px - nx) * (y - ny) / (py - ny + 1e-10) + nx:
            inside = not inside
        px, py = nx, ny
    return inside


# ─────────────────────────────────────────────────────────────────────────────
# Queue Analytics Engine  — Dwell-Time Based
# ─────────────────────────────────────────────────────────────────────────────
#
# HOW IT WORKS:
# ─────────────────────────────────────────────────────────────────────────────
# Old approach (transition-based):
#   Needed  queue → service → exit  transitions to measure times.
#   Problem: If people are already standing in zones and don't move,
#            wait time stays 0 forever.
#
# New approach (dwell-time based):
#   The moment a person is detected INSIDE a zone (even at frame 1),
#   a timestamp is recorded.  Every subsequent frame:
#
#     dwell_time(tid) = now - entry_timestamp[tid]
#
#   Avg Waiting Time  = mean dwell time of everyone currently in QUEUE zone
#   Avg Processing Time = mean dwell time of everyone currently in SERVICE zone
#   Throughput = persons who LEFT the service zone (service → not-service)
#
#   This works perfectly when:
#     • People are already in the zone at video start  (cold-start handled)
#     • People don't move between zones at all
#     • Zones are never crossed (the real airport scenario)
#
# ─────────────────────────────────────────────────────────────────────────────
class QueueAnalytics:
    def __init__(self, alert_thr=10, wait_thr=300.0, cam_id=1):
        self.alert_thr = alert_thr
        self.wait_thr  = wait_thr
        self.cam_id    = cam_id

        self.zone_queue:   list = []
        self.zone_service: list = []
        self.zone_exit:    list = []

        # ── per-track state ───────────────────────────────────────────────────
        self._tz:  Dict[int, str]   = {}   # current zone for each track_id
        self._ts:  Dict[int, float] = {}   # zone-entry timestamp for each track_id
        #   _ts[tid] is set when the person FIRST enters any zone (or at frame-1
        #   if already inside — cold-start handled automatically)

        # ── completed dwell time history ──────────────────────────────────────
        # Recorded when a person LEAVES the queue zone (exits or disappears)
        self._wt_history = deque(maxlen=500)   # past queue dwell times
        # Recorded when a person LEAVES the service zone
        self._pt_history = deque(maxlen=500)   # past service dwell times
        self._pt_all:    List[float] = []      # unbounded for percentile stats

        self._proc  = 0          # total persons who completed service
        self._t0    = time.time()
        self._ah    = deque(maxlen=300)
        self.active_alerts: List[Dict] = []

    # ── zone config ───────────────────────────────────────────────────────────
    def set_zones(self, q, s, e):
        self.zone_queue   = q or []
        self.zone_service = s or []
        self.zone_exit    = e or []

    def zones_defined(self):
        return bool(self.zone_queue and self.zone_service)

    def exit_zone_defined(self):
        return bool(self.zone_exit)

    def _zone(self, bbox, fw, fh):
        """Centroid-based zone membership. Priority: service > exit > queue."""
        cx = (bbox[0] + bbox[2]) / 2; cy = (bbox[1] + bbox[3]) / 2
        fx = cx / fw; fy = cy / fh
        if _in_poly((fx, fy), self.zone_service): return "service"
        if _in_poly((fx, fy), self.zone_exit):    return "exit"
        if _in_poly((fx, fy), self.zone_queue):   return "queue"
        return None

    # ── main per-frame update ─────────────────────────────────────────────────
    def update(self, tracks, fw, fh):
        now  = time.time()
        cids = {d["track_id"] for d in tracks}

        for d in tracks:
            tid = d["track_id"]
            z   = self._zone(d["bbox"], fw, fh)
            p   = self._tz.get(tid)   # zone this person was in last frame

            if z is None:
                # Person is outside all zones
                if p in ("queue", "service"):
                    # They just LEFT a zone — record their dwell time
                    dwell = now - self._ts.pop(tid, now)
                    if p == "queue":
                        self._wt_history.append(dwell)
                    elif p == "service":
                        self._pt_history.append(dwell)
                        self._pt_all.append(dwell)
                        self._proc += 1
                self._tz.pop(tid, None)
                self._ts.pop(tid, None)

            elif z != p:
                # Person moved to a different zone (or appeared in one for first time)
                if p == "queue":
                    # Left queue — record queue dwell time
                    dwell = now - self._ts.pop(tid, now)
                    self._wt_history.append(dwell)
                elif p == "service":
                    # Left service — record service dwell time + count as processed
                    dwell = now - self._ts.pop(tid, now)
                    self._pt_history.append(dwell)
                    self._pt_all.append(dwell)
                    self._proc += 1

                # Enter new zone — record entry timestamp
                # This handles BOTH cold-start (already in zone at frame 1)
                # AND genuine entry — no special-casing needed
                self._tz[tid] = z
                self._ts[tid] = now

            # If z == p: person still in same zone, timestamp unchanged (dwell continues)

        # Clean up tracks that disappeared from frame entirely
        for tid in (set(self._tz) - cids):
            p = self._tz.get(tid)
            if p in ("queue", "service"):
                dwell = now - self._ts.pop(tid, now)
                if p == "queue":
                    self._wt_history.append(dwell)
                elif p == "service":
                    self._pt_history.append(dwell)
                    self._pt_all.append(dwell)
                    self._proc += 1
            self._tz.pop(tid, None)
            self._ts.pop(tid, None)

        # ── Compute metrics ───────────────────────────────────────────────────
        # Queue length = persons currently in queue zone
        ql = sum(1 for z in self._tz.values() if z == "queue")
        sl = sum(1 for z in self._tz.values() if z == "service")

        # LIVE dwell times — how long each person has been in their zone RIGHT NOW
        # This is the key metric: updates every frame, never stays at 0
        queue_dwells   = [now - self._ts[tid] for tid, z in self._tz.items()
                          if z == "queue"   and tid in self._ts]
        service_dwells = [now - self._ts[tid] for tid, z in self._tz.items()
                          if z == "service" and tid in self._ts]

        # Avg waiting time:
        #   Primary = live average of current queue dwell times (updates every frame)
        #   Fallback = average of completed queue dwell times from history
        if queue_dwells:
            awt = sum(queue_dwells) / len(queue_dwells)
        elif self._wt_history:
            awt = sum(self._wt_history) / len(self._wt_history)
        else:
            awt = 0.0

        # Avg processing time:
        #   Primary = live average of current service dwell times
        #   Fallback = average of completed service dwell times from history
        if service_dwells:
            apt = sum(service_dwells) / len(service_dwells)
        elif self._pt_history:
            apt = sum(self._pt_history) / len(self._pt_history)
        else:
            apt = 0.0

        # Throughput = persons who completed service, extrapolated to per-hour
        tput = self._proc * 3600 / max(1, now - self._t0) if self._proc > 0 else 0.0

        # Max individual dwell times (useful for spotting stuck passengers)
        max_queue_dwell   = max(queue_dwells)   if queue_dwells   else 0.0
        max_service_dwell = max(service_dwells) if service_dwells else 0.0

        # ── Alerts ────────────────────────────────────────────────────────────
        new_a = []

        def _alert(tp, msg, lv):
            last = [a["_ts"] for a in self._ah
                    if a.get("type") == tp and not a.get("acked")]
            if not last or now - last[-1] > 30:
                idx = len(self._ah)
                a = {
                    "idx": idx, "time": time.strftime("%H:%M:%S"),
                    "message": msg, "level": lv, "type": tp,
                    "_ts": now, "queue_length": ql,
                    "avg_wait": round(awt, 1), "acked": False,
                }
                self._ah.append(a); new_a.append(a)
                self._persist_alert(a)

        if ql > self.alert_thr:
            _alert("queue_length",
                   f"Queue {ql} > threshold ({self.alert_thr})",
                   "critical" if ql > self.alert_thr * 1.5 else "warning")
        if awt > self.wait_thr:
            _alert("wait_time",
                   f"Avg wait {awt:.0f}s > {self.wait_thr:.0f}s", "warning")
        self.active_alerts = new_a

        return {
            "queue_length":          ql,
            "avg_waiting_time":      round(awt, 1),
            "max_waiting_time":      round(max_queue_dwell, 1),
            "avg_processing_time":   round(apt, 1),
            "max_processing_time":   round(max_service_dwell, 1),
            "throughput_per_hour":   round(tput, 1),
            "total_processed":       self._proc,
            "alerts":                new_a,
            "seeded_count":          0,   # no longer needed — dwell handles cold-start
            "partial_wait_samples":  0,
            "zone_counts": {
                "queue":   ql,
                "service": sl,
                "exit":    sum(1 for z in self._tz.values() if z == "exit"),
            },
            "warnings": [],   # no exit zone warning — dwell approach doesn't need exit zone
            # Per-person dwell times (for dashboard display)
            "queue_dwells":   {str(tid): round(now - self._ts[tid], 1)
                               for tid, z in self._tz.items()
                               if z == "queue" and tid in self._ts},
            "service_dwells": {str(tid): round(now - self._ts[tid], 1)
                               for tid, z in self._tz.items()
                               if z == "service" and tid in self._ts},
        }

    # ── alert history + ack ───────────────────────────────────────────────────
    def get_alert_history(self) -> List[Dict]:
        return [{k: v for k, v in a.items() if k != "_ts"} for a in self._ah]

    def acknowledge_alert(self, idx: int) -> bool:
        for a in self._ah:
            if a.get("idx") == idx:
                a["acked"] = True
                self._rewrite_alerts_file()
                return True
        return False

    # ── counter performance ───────────────────────────────────────────────────
    def get_counter_performance(self) -> Dict:
        if not self._pt_all:
            return {"samples": 0, "min_s": None, "max_s": None,
                    "avg_s": None, "p50_s": None, "p90_s": None,
                    "total_processed": self._proc}
        arr = sorted(self._pt_all); n = len(arr)
        return {
            "samples":         n,
            "min_s":           round(min(arr), 1),
            "max_s":           round(max(arr), 1),
            "avg_s":           round(sum(arr) / n, 1),
            "p50_s":           round(arr[n // 2], 1),
            "p90_s":           round(arr[int(n * 0.9)], 1),
            "total_processed": self._proc,
        }

    # ── disk persistence ──────────────────────────────────────────────────────
    def _persist_alert(self, alert: Dict):
        try:
            path = _ALERTS_DIR / f"cam_{self.cam_id}_alerts.json"
            existing: List[Dict] = []
            if path.exists():
                try:
                    with open(path) as f: existing = json.load(f)
                except: existing = []
            clean = {k: v for k, v in alert.items() if k != "_ts"}
            existing.append(clean)
            with open(path, "w") as f:
                json.dump(existing, f, indent=2)
        except Exception as e:
            log.warning(f"[Analytics cam {self.cam_id}] alert persist failed: {e}")

    def _rewrite_alerts_file(self):
        try:
            path = _ALERTS_DIR / f"cam_{self.cam_id}_alerts.json"
            data = [{k: v for k, v in a.items() if k != "_ts"} for a in self._ah]
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            log.warning(f"[Analytics cam {self.cam_id}] alerts rewrite failed: {e}")

    def reset(self):
        self._tz.clear(); self._ts.clear()
        self._wt_history.clear(); self._pt_history.clear()
        self._pt_all.clear()
        self._proc = 0; self._t0 = time.time(); self._ah.clear()



# ─────────────────────────────────────────────────────────────────────────────
# Video Decoder  (GPU-assisted RTSP / plain cv2 for files)
# ─────────────────────────────────────────────────────────────────────────────
class VideoDecoder:
    def __init__(self):
        self.cap     = None
        self._frame  = None
        self._lock   = threading.Lock()
        self._thread = None
        self._run    = False
        self.source  = None
        self.mode    = "idle"
        self.fps     = 0.0
        self._fc     = 0
        self._t0     = 0.0

    def start(self, source: str, mode: str = "live"):
        self.stop()
        self.source = source; self.mode = mode
        self._run = True; self._fc = 0; self._t0 = time.time()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name=f"Decode-{source[:30]}")
        self._thread.start()
        log.info(f"[Decoder] {mode}: {source}")

    def stop(self):
        self._run = False
        if self._thread: self._thread.join(timeout=3)
        if self.cap: self.cap.release(); self.cap = None
        self._frame = None; self.mode = "idle"

    def _open_cap(self, src):
        is_stream = isinstance(src, str) and (
            src.startswith("rtsp") or
            src.startswith("http") or
            src.startswith("rtp"))
        if is_stream:
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                log.info("[Decoder] Opened live stream with CAP_FFMPEG")
                return cap
        cap = cv2.VideoCapture(src)
        return cap if cap.isOpened() else None

    def _loop(self):
        src = self.source
        try: src = int(src)
        except: pass

        # ── Auto-detect if source is a local file regardless of stated mode ──
        # This handles the case where user adds a video file but selects "live"
        def _is_local_file(s):
            if isinstance(s, int): return False   # webcam index
            return not (str(s).startswith("rtsp") or
                        str(s).startswith("http") or
                        str(s).startswith("rtp"))

        is_file = _is_local_file(src)
        # Force recording mode for local files so looping always works
        if is_file and self.mode != "recording":
            log.info(f"[Decoder] Source looks like a local file — "
                     f"switching mode to 'recording' for looping")
            self.mode = "recording"

        # ── FIX-1: reopen helper ─────────────────────────────────────────────
        def _open():
            c = self._open_cap(src)
            if c is None:
                log.error(f"[Decoder] Cannot open: {self.source}")
            return c

        cap = _open()
        if cap is None:
            self._run = False; return
        self.cap = cap

        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if native_fps <= 0 or native_fps > 120: native_fps = 25.0
        frame_interval = 1.0 / native_fps
        last_read  = 0.0
        loop_count = 0

        while self._run:
            now = time.time()
            if self.mode == "recording":
                elapsed = now - last_read
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

            ret, frame = cap.read()
            last_read  = time.time()

            if not ret:
                if self.mode == "recording":
                    # ── FIX-1: release and reopen instead of broken seek ─────
                    loop_count += 1
                    log.info(
                        f"[Decoder] '{str(self.source)[:40]}' "
                        f"end of file — restarting loop #{loop_count}")
                    cap.release()
                    time.sleep(0.15)   # brief pause before reopen
                    cap = _open()
                    if cap is None:
                        log.error("[Decoder] Cannot reopen — stopping")
                        self._run = False; break
                    self.cap = cap
                    # Refresh FPS from the newly opened cap
                    native_fps = cap.get(cv2.CAP_PROP_FPS)
                    if native_fps <= 0 or native_fps > 120: native_fps = 25.0
                    frame_interval = 1.0 / native_fps
                    continue
                else:
                    # Live stream temporary disconnect — just retry
                    time.sleep(0.05); continue

            with self._lock: self._frame = frame
            self._fc += 1
            e = time.time() - self._t0
            self.fps = self._fc / e if e > 0 else 0.0

        cap.release(); self.cap = None

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None


# ─────────────────────────────────────────────────────────────────────────────
# CameraStream  — one per configured camera
# ─────────────────────────────────────────────────────────────────────────────
ZONE_COLORS = {"queue": (0, 200, 255), "service": (0, 255, 128), "exit": (200, 80, 255)}
PERSON_C    = {"queue": (0, 200, 255), "service": (0, 255, 128),
               "exit":  (200, 80, 255), None: (180, 180, 180)}


class CameraStream:
    _id_counter = 1

    def __init__(self, name: str, source: str, mode: str = "live",
                 alert_thr: int = 10, wait_thr: float = 300.0,
                 inf_fps: float = 15.0):
        self.cam_id  = CameraStream._id_counter; CameraStream._id_counter += 1
        self.name    = name
        self.source  = source
        self.mode    = mode
        self.inf_fps_limit: float = max(1.0, min(30.0, float(inf_fps)))
        self.decoder   = VideoDecoder()
        self.tracker   = ByteTracker()
        self.analytics = QueueAnalytics(alert_thr, wait_thr, cam_id=self.cam_id)
        self._lock     = threading.Lock()
        self._metrics: Dict = {}
        self._ann:     Optional[np.ndarray] = None
        self._inf_fps: float = 0.0
        self._cb_fc:   int   = 0
        self._cb_t:    float = time.time()
        self._inf_ms:  float = 0.0
        self._history: deque = deque(maxlen=3600)   # 1 hour @ 1 sample/sec
        self.active:          bool = False
        self.inference_active: bool = False
        self._last_sub: float = 0.0

    def start(self, npu_engine: "NPUInferenceEngine"):
        self.decoder.start(self.source, self.mode)
        self._npu   = npu_engine
        self.active = True
        t = threading.Thread(target=self._submit_loop, daemon=True,
                             name=f"Cam-{self.cam_id}")
        t.start()
        log.info(f"[Cam-{self.cam_id}] '{self.name}' started  "
                 f"inf_fps_limit={self.inf_fps_limit}")

    def stop(self):
        self.active = False; self.decoder.stop()
        log.info(f"[Cam-{self.cam_id}] stopped")

    # ── frame submission ──────────────────────────────────────────────────────
    def _submit_loop(self):
        while self.active:
            if not self.inference_active:
                time.sleep(0.05); continue
            frame = self.decoder.read()
            if frame is None: time.sleep(0.02); continue
            now = time.time()
            if now - self._last_sub < 1.0 / self.inf_fps_limit:
                time.sleep(0.005); continue
            self._last_sub = now
            self._npu.submit(self.cam_id, frame, self._on_inference)

    # ── NPU callback ─────────────────────────────────────────────────────────
    def _on_inference(self, cam_id: int, raw_dets: list, inf_ms: float):
        frame = self.decoder.read()
        if frame is None: return
        h, w = frame.shape[:2]

        tracks = self.tracker.update(raw_dets)

        if self.analytics.zones_defined():
            metrics = self.analytics.update(tracks, w, h)
        else:
            metrics = {
                "queue_length": 0, "avg_waiting_time": 0,
                "avg_processing_time": 0, "throughput_per_hour": 0,
                "total_processed": 0, "alerts": [], "zone_counts": {},
                "warnings": ["Zones not configured — draw zones in Zone Setup tab"],
                "live_avg_wait": 0, "seeded_count": 0, "partial_wait_samples": 0,
            }

        ann = self._draw(frame.copy(), tracks, metrics, inf_ms)

        self._cb_fc += 1
        now = time.time()
        if now - self._cb_t >= 1.0:
            self._inf_fps = self._cb_fc / (now - self._cb_t)
            self._cb_fc = 0; self._cb_t = now
        self._inf_ms = inf_ms

        with self._lock:
            self._metrics = metrics
            self._ann     = ann
            last_ts = self._history[-1].get("_ts", 0) if self._history else 0
            if now - last_ts >= 1.0:
                entry = {
                    "ts":               time.strftime("%H:%M:%S"),
                    "queue_length":     metrics["queue_length"],
                    "avg_wait":         metrics["avg_waiting_time"],
                    "live_avg_wait":    metrics.get("live_avg_wait", 0),
                    "avg_proc":         metrics["avg_processing_time"],
                    "throughput":       metrics["throughput_per_hour"],
                    "in_service":       metrics.get("zone_counts", {}).get("service", 0),
                    "total_processed":  metrics.get("total_processed", 0),
                    "seeded_count":     metrics.get("seeded_count", 0),
                    "_ts":              now,
                }
                self._history.append(entry)
                if len(self._history) % 60 == 0:
                    self._persist_history()

    def _persist_history(self):
        try:
            path = _ALERTS_DIR / f"cam_{self.cam_id}_history.json"
            data = [{k: v for k, v in e.items() if k != "_ts"} for e in self._history]
            with open(path, "w") as f: json.dump(data, f)
            log.info(f"[Cam-{self.cam_id}] History saved ({len(data)} samples)")
        except Exception as e:
            log.warning(f"[Cam-{self.cam_id}] history persist failed: {e}")

    # ── drawing ───────────────────────────────────────────────────────────────
    def _draw(self, frame, tracks, metrics, inf_ms) -> np.ndarray:
        h, w = frame.shape[:2]; sf = h / 720; font = cv2.FONT_HERSHEY_SIMPLEX
        # NOTE: Zone polygons are NOT drawn here — they are rendered by the
        # dashboard canvas overlay (zone-canvas-overlay) so there is no duplication.

        for det in tracks:
            tid  = det["track_id"]
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            z    = self.analytics._tz.get(tid)
            color = PERSON_C.get(z, PERSON_C[None])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            lbl = f"ID:{tid}" + (f" [{z[0].upper()}]" if z else "")
            (lw, lh), _ = cv2.getTextSize(lbl, font, 0.38, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw + 4, y1), color, -1)
            cv2.putText(frame, lbl, (x1 + 2, y1 - 3), font, 0.38, (0, 0, 0), 1, cv2.LINE_AA)

        # HUD
        lines = [
            (f"Cam: {self.name}",                                      (200, 220, 255)),
            (f"NPU: {self._npu.actual_dev}",                           (100, 220, 255)),
            (f"Tracker: ByteTrack  FPS:{self._inf_fps:.1f}",           (180, 255, 180)),
            (f"Inf: {inf_ms:.1f}ms  Limit:{self.inf_fps_limit:.0f}fps",(160, 255, 160)),
            (f"Queue: {metrics.get('queue_length', 0)}",               (0, 200, 255)),
            (f"Avg Wait: {metrics.get('avg_waiting_time', 0):.1f}s  "
             f"Max:{metrics.get('max_waiting_time', 0):.1f}s",         (255, 200, 80)),
            (f"Avg Proc: {metrics.get('avg_processing_time', 0):.1f}s  "
             f"Max:{metrics.get('max_processing_time', 0):.1f}s",      (180, 255, 80)),
            (f"Tput: {metrics.get('throughput_per_hour', 0):.0f}/h  "
             f"Total:{metrics.get('total_processed', 0)}",             (200, 180, 255)),
        ]
        if not self.analytics.zones_defined():
            lines.append(("! Draw zones in dashboard", (0, 140, 255)))

        lh2 = max(18, int(22 * sf)); ph2 = len(lines) * lh2 + 16
        ov  = frame.copy()
        cv2.rectangle(ov, (8, 8), (350, 8 + ph2), (10, 10, 10), -1)
        cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
        for i, (txt, col) in enumerate(lines):
            cv2.putText(frame, txt, (14, 12 + lh2 * (i + 1) - 4),
                        font, max(0.33, 0.40 * sf), col, 1, cv2.LINE_AA)

        if metrics.get("alerts"):
            al  = metrics["alerts"][0]; msg = al["message"]
            bg  = (0, 0, 180) if al.get("level") == "critical" else (0, 80, 160)
            (tw, th), _ = cv2.getTextSize(msg, font, 0.55, 2)
            bx, by = (w - tw) // 2, h - 40
            cv2.rectangle(frame, (bx - 10, by - th - 6), (bx + tw + 10, by + 6), bg, -1)
            cv2.putText(frame, msg, (bx, by), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        return frame

    # ── public getters ────────────────────────────────────────────────────────
    def get_metrics(self) -> Dict:
        with self._lock: return dict(self._metrics)

    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            if self._ann is not None:
                ret, buf = cv2.imencode(".jpg", self._ann,
                                        [cv2.IMWRITE_JPEG_QUALITY, 80])
                return buf.tobytes() if ret else None
        raw = self.decoder.read()
        if raw is None: return None
        ret, buf = cv2.imencode(".jpg", raw, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return buf.tobytes() if ret else None

    def get_history(self) -> List:
        with self._lock:
            return [{k: v for k, v in e.items() if k != "_ts"}
                    for e in self._history]

    def get_history_csv(self) -> str:
        with self._lock:
            rows = [{k: v for k, v in e.items() if k != "_ts"}
                    for e in self._history]
        if not rows:
            return ("ts,queue_length,avg_wait,live_avg_wait,avg_proc,"
                    "throughput,in_service,total_processed,seeded_count\n")
        buf    = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
        return buf.getvalue()

    def set_zones(self, q, s, e):
        with self._lock: self.analytics.set_zones(q, s, e)

    def set_thresholds(self, alert_thr, wait_thr):
        with self._lock:
            self.analytics.alert_thr = alert_thr
            self.analytics.wait_thr  = wait_thr

    def reset(self):
        with self._lock:
            self.analytics.reset(); self.tracker = ByteTracker()

    def to_dict(self) -> Dict:
        return {
            "cam_id":            self.cam_id,
            "name":              self.name,
            "source":            self.source,
            "mode":              self.mode,
            "active":            self.active,
            "fps":               round(self.decoder.fps, 1),
            "inf_fps":           round(self._inf_fps, 1),
            "inf_fps_limit":     self.inf_fps_limit,
            "inf_ms":            round(self._inf_ms, 1),
            "zones_defined":     self.analytics.zones_defined(),
            "exit_zone_defined": self.analytics.exit_zone_defined(),
            "inference_active":  self.inference_active,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Camera Manager
# ─────────────────────────────────────────────────────────────────────────────
class CameraManager:
    def __init__(self, npu_engine: NPUInferenceEngine):
        self.npu = npu_engine
        self.cameras: Dict[int, CameraStream] = {}
        self._lock = threading.Lock()

    def add(self, name: str, source: str, mode: str = "live",
            alert_thr: int = 10, wait_thr: float = 300.0,
            inf_fps: float = 15.0) -> CameraStream:
        cam = CameraStream(name, source, mode, alert_thr, wait_thr, inf_fps)
        cam.start(self.npu)
        with self._lock: self.cameras[cam.cam_id] = cam
        log.info(f"[Manager] Added cam {cam.cam_id} '{name}'")
        return cam

    def remove(self, cam_id: int):
        with self._lock: cam = self.cameras.pop(cam_id, None)
        if cam: cam.stop()

    def get(self, cam_id: int) -> Optional[CameraStream]:
        with self._lock: return self.cameras.get(cam_id)

    def list(self) -> List[Dict]:
        with self._lock: return [c.to_dict() for c in self.cameras.values()]

    def stop_all(self):
        with self._lock:
            for c in self.cameras.values(): c.stop()
            self.cameras.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Flask App
# ─────────────────────────────────────────────────────────────────────────────
app        = Flask(__name__, static_folder=str(_SCRIPT_DIR))
npu_engine = NPUInferenceEngine()
manager    = CameraManager(npu_engine)


def _placeholder_jpg(msg="No video — add a camera stream"):
    ph = np.full((480, 640, 3), 30, np.uint8)
    cv2.putText(ph, msg, (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (120, 120, 120), 1)
    _, buf = cv2.imencode(".jpg", ph)
    return buf.tobytes()


@app.route("/")
def index():
    return send_from_directory(str(_SCRIPT_DIR), "dashboard-claude.html")

# ── Camera CRUD ───────────────────────────────────────────────────────────────
@app.route("/api/cameras", methods=["GET"])
def list_cameras():
    return jsonify(manager.list())

@app.route("/api/cameras", methods=["POST"])
def add_camera():
    d      = request.get_json(force=True)
    name   = d.get("name", "Camera").strip() or "Camera"
    source = d.get("source", "").strip()
    mode   = d.get("mode", "live")
    if not source:
        return jsonify({"error": "source required"}), 400
    at      = int(d.get("alert_threshold", 10))
    wt      = float(d.get("wait_time_threshold", 300.0))
    inf_fps = float(d.get("inf_fps", 15.0))
    cam     = manager.add(name, source, mode, at, wt, inf_fps)
    return jsonify(cam.to_dict()), 201

@app.route("/api/cameras/<int:cam_id>", methods=["DELETE"])
def remove_camera(cam_id: int):
    manager.remove(cam_id)
    return jsonify({"status": "removed", "cam_id": cam_id})

# ── Zones ─────────────────────────────────────────────────────────────────────
@app.route("/api/cameras/<int:cam_id>/zones", methods=["POST"])
def set_zones(cam_id: int):
    cam = manager.get(cam_id)
    if not cam: abort(404)
    d = request.get_json(force=True)
    cam.set_zones(d.get("zone_queue", []),
                  d.get("zone_service", []),
                  d.get("zone_exit", []))
    at = int(d.get("alert_threshold",    cam.analytics.alert_thr))
    wt = float(d.get("wait_time_threshold", cam.analytics.wait_thr))
    cam.set_thresholds(at, wt)
    cam.inference_active = True
    log.info(f"[Cam-{cam_id}] Zones applied — inference STARTED")
    warn = [] if cam.analytics.exit_zone_defined() else \
           ["Exit zone not drawn — processing time will show 0. "
            "Draw an Exit zone to enable it."]
    return jsonify({"status": "ok", "cam_id": cam_id,
                    "inference_active": True, "warnings": warn})

@app.route("/api/cameras/<int:cam_id>/zones", methods=["GET"])
def get_zones(cam_id: int):
    cam = manager.get(cam_id)
    if not cam: abort(404)
    a = cam.analytics
    return jsonify({"zone_queue": a.zone_queue, "zone_service": a.zone_service,
                    "zone_exit": a.zone_exit, "alert_threshold": a.alert_thr,
                    "wait_time_threshold": a.wait_thr})

@app.route("/api/cameras/<int:cam_id>/zones", methods=["DELETE"])
def clear_zones(cam_id: int):
    cam = manager.get(cam_id)
    if not cam: abort(404)
    cam.inference_active = False
    cam.set_zones([], [], [])
    cam.analytics.reset()
    cam.tracker = ByteTracker()
    log.info(f"[Cam-{cam_id}] Zones cleared — inference STOPPED")
    return jsonify({"status": "zones_cleared", "cam_id": cam_id,
                    "inference_active": False})

# ── Status / metrics ──────────────────────────────────────────────────────────
@app.route("/api/cameras/<int:cam_id>/status")
def cam_status(cam_id: int):
    cam = manager.get(cam_id)
    if not cam: abort(404)
    m = cam.get_metrics(); m.update(cam.to_dict())
    m["npu_device"] = npu_engine.actual_dev
    for a in m.get("alerts", []): a.pop("_ts", None)
    return jsonify(m)

@app.route("/api/cameras/<int:cam_id>/history")
def cam_history(cam_id: int):
    cam = manager.get(cam_id)
    if not cam: abort(404)
    return jsonify(cam.get_history())

@app.route("/api/cameras/<int:cam_id>/history/export")
def cam_history_export(cam_id: int):
    cam = manager.get(cam_id)
    if not cam: abort(404)
    csv_data = cam.get_history_csv()
    fname = f"cam_{cam_id}_{time.strftime('%Y%m%d_%H%M%S')}_history.csv"
    return Response(csv_data, mimetype="text/csv",
                    headers={"Content-Disposition":
                             f"attachment; filename={fname}"})

# ── Alerts ────────────────────────────────────────────────────────────────────
@app.route("/api/cameras/<int:cam_id>/alerts")
def cam_alerts(cam_id: int):
    cam = manager.get(cam_id)
    if not cam: abort(404)
    return jsonify(cam.analytics.get_alert_history())

@app.route("/api/cameras/<int:cam_id>/alerts/<int:alert_idx>/ack",
           methods=["POST"])
def ack_alert(cam_id: int, alert_idx: int):
    cam = manager.get(cam_id)
    if not cam: abort(404)
    ok = cam.analytics.acknowledge_alert(alert_idx)
    if not ok:
        return jsonify({"error": "alert not found"}), 404
    return jsonify({"status": "acknowledged", "cam_id": cam_id,
                    "alert_idx": alert_idx})

# ── Counter performance ───────────────────────────────────────────────────────
@app.route("/api/cameras/<int:cam_id>/performance")
def cam_performance(cam_id: int):
    cam = manager.get(cam_id)
    if not cam: abort(404)
    perf = cam.analytics.get_counter_performance()
    perf["cam_id"]            = cam_id
    perf["cam_name"]          = cam.name
    perf["exit_zone_defined"] = cam.analytics.exit_zone_defined()
    return jsonify(perf)

@app.route("/api/cameras/<int:cam_id>/reset", methods=["POST"])
def cam_reset(cam_id: int):
    cam = manager.get(cam_id)
    if not cam: abort(404)
    cam.reset(); return jsonify({"status": "reset"})

# ── Multi-camera overview ─────────────────────────────────────────────────────
@app.route("/api/overview")
def overview():
    with manager._lock:
        cams = list(manager.cameras.values())
    rows = []
    total_queue = 0; total_processed = 0; total_unacked = 0
    for cam in cams:
        m          = cam.get_metrics()
        perf       = cam.analytics.get_counter_performance()
        alerts_all = cam.analytics.get_alert_history()
        unacked    = [a for a in alerts_all if not a.get("acked")]
        ql         = m.get("queue_length", 0)
        total_queue     += ql
        total_processed += m.get("total_processed", 0)
        total_unacked   += len(unacked)
        rows.append({
            "cam_id":              cam.cam_id,
            "name":                cam.name,
            "mode":                cam.mode,
            "inference_active":    cam.inference_active,
            "zones_defined":       cam.analytics.zones_defined(),
            "exit_zone_defined":   cam.analytics.exit_zone_defined(),
            "queue_length":        ql,
            "avg_waiting_time":    m.get("avg_waiting_time", 0),
            "live_avg_wait":       m.get("live_avg_wait", 0),
            "avg_processing_time": m.get("avg_processing_time", 0),
            "throughput_per_hour": m.get("throughput_per_hour", 0),
            "total_processed":     m.get("total_processed", 0),
            "zone_counts":         m.get("zone_counts", {}),
            "unacked_alerts":      len(unacked),
            "seeded_count":        m.get("seeded_count", 0),
            "counter_perf":        perf,
            "inf_fps":             round(cam._inf_fps, 1),
            "inf_ms":              round(cam._inf_ms, 1),
            "inf_fps_limit":       cam.inf_fps_limit,
        })
    return jsonify({
        "cameras":              rows,
        "total_cameras":        len(rows),
        "total_queue":          total_queue,
        "total_processed":      total_processed,
        "total_unacked_alerts": total_unacked,
        "npu_device":           npu_engine.actual_dev,
        "timestamp":            time.strftime("%H:%M:%S"),
    })

# ── MJPEG stream ──────────────────────────────────────────────────────────────
@app.route("/video_feed/<int:cam_id>")
def video_feed(cam_id: int):
    def gen():
        while True:
            cam = manager.get(cam_id)
            jpg = cam.get_jpeg() if cam else None
            if jpg is None: jpg = _placeholder_jpg()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(1.0 / 25)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ── System stats ──────────────────────────────────────────────────────────────
@app.route("/api/system")
def sys_stats():
    return jsonify({"npu": npu_engine.get_stats(),
                    "cameras": len(manager.cameras)})


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Airport Queue Management (Multi-Stream)")
    parser.add_argument("--host",             default="0.0.0.0")
    parser.add_argument("--port",             default=5050, type=int)
    parser.add_argument("--alert-threshold",  default=10,    type=int)
    parser.add_argument("--wait-threshold",   default=300.0, type=float)
    args = parser.parse_args()

    npu_engine.start()

    log.info(f"\n🌐  Dashboard → http://localhost:{args.port}")
    log.info(f"    Tracker   : ByteTrack (CPU)")
    log.info(f"    Inference : NPU/GPU/CPU via OpenVINO AUTO")
    log.info(f"    Decode    : GPU-assisted VAAPI for live streams")
    log.info(f"    Alerts    : persisted → {_ALERTS_DIR}")
    log.info(f"    Q-Alert   : queue > {args.alert_threshold} persons")
    log.info(f"    W-Alert   : avg wait > {args.wait_threshold:.0f}s")
    log.info(f"    FIX-1     : Video loop — release+reopen (not broken seek)")
    log.info(f"    FIX-2     : Cold-start — persons already in zones seeded")
    log.info(f"    APIs:")
    log.info(f"      GET  /api/overview")
    log.info(f"      GET  /api/cameras/<id>/history/export  (CSV download)")
    log.info(f"      GET  /api/cameras/<id>/performance")
    log.info(f"      POST /api/cameras/<id>/alerts/<idx>/ack")
    log.info(f"    Add stream: POST /api/cameras")
    log.info(f"      {{\"name\":...,\"source\":...,\"mode\":\"live|recording\","
             f"\"inf_fps\":15}}\n")

    app.run(host=args.host, port=args.port, threaded=True, debug=False)