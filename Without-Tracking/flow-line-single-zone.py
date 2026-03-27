"""
Airport Queue Management System — Multi-Stream Edition
=======================================================
Hardware pipeline (Intel Core Ultra 7 255H):
  • NPU  → YOLOv8n INT8 inference (OpenVINO worker threads, shared queue)
  • iGPU → Video decode via QSV / VA-API (Intel GPU hardware decode)
           Works for BOTH RTSP live streams AND recorded video files
  • CPU  → Flask REST API, analytics, drawing

GPU Decode Priority:
  1. Intel QSV (Quick Sync Video) — h264_qsv / hevc_qsv
  2. Intel VA-API                 — h264_vaapi
  3. FFmpeg auto hwaccel          — hwaccel=auto
  4. CPU software decode          — fallback

Architecture:
  Camera/File → VideoDecoder (iGPU QSV/VAAPI) → frame_queue
                → NPUInferenceEngine (worker pool) → callback
                → QueueAnalytics (zone-change debounce, CPU)
                → draw overlay → MJPEG stream / REST API → Dashboard

Zones (drawn interactively in the dashboard, stored as normalised [0..1] polygon):
  • Queue Zone   – passengers waiting
  • Service Zone – passengers at counter
  • Exit Zone    – passengers leaving (optional — dwell-time approach works without it)

Metrics (per stream) — Dwell-Time Based:
  Queue Length · Avg Waiting Time · Avg Processing Time · Throughput/hr

  Dwell-time approach: timestamps recorded on first detection inside zone.
  avg_wait = mean(now - entry_ts) for all persons in queue zone — RIGHT NOW.
  Works even if persons never move between zones.

Run:
  python queue_system.py [--host 0.0.0.0] [--port 5050]

Install GPU decode support (Ubuntu):
  sudo apt install intel-media-va-driver-non-free vainfo libva-drm2 ffmpeg
  sudo usermod -aG video,render $USER  # then re-login

Add streams:
  POST /api/cameras  {"name":"Gate-1","source":"rtsp://...","mode":"live","inf_fps":15}
  POST /api/cameras  {"name":"Counter-A","source":"/path/video.mp4","mode":"recording"}
"""

import os, sys, time, json, csv, threading, queue, argparse, logging, io, subprocess
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple
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
    _SCRIPT_DIR / "best_int8_openvino_model"     / "best.xml",   # ← your trained model FIRST
    _SCRIPT_DIR / "yolov8n_int8_openvino_model"  / "yolov8n.xml",  # fallback
    _LOITER_DIR / "yolov8n_int8_openvino_model"  / "yolov8n.xml",
    _LOITER_DIR / "yolov8n_openvino_model"        / "yolov8n.xml",
]

CONF_THR = 0.12
IOU_THR  = 0.30
INF_SIZE = 640

log.info("=" * 68)
log.info("  AIRPORT QUEUE MANAGEMENT — Intel Core Ultra 7 (NPU+iGPU+CPU)")
log.info("=" * 68)


# ─────────────────────────────────────────────────────────────────────────────
# GPU Decode capability detection (run once at startup)
# ─────────────────────────────────────────────────────────────────────────────
def _detect_gpu_decode() -> Tuple[bool, bool, str]:
    """
    Detect Intel GPU hardware decode capabilities.
    Returns: (vaapi_ok, qsv_ok, render_device)
    """
    render_device = "/dev/dri/renderD128"

    # Find the correct render device
    try:
        dri = Path("/dev/dri")
        if dri.exists():
            render_nodes = sorted(dri.glob("renderD*"))
            if render_nodes:
                render_device = str(render_nodes[0])
    except Exception:
        pass

    # Check VA-API
    vaapi_ok = False
    try:
        r = subprocess.run(
            ["vainfo", "--display", "drm", "--device", render_device],
            capture_output=True, text=True, timeout=5)
        vaapi_ok = "VAEntrypointVLD" in r.stdout
        if vaapi_ok:
            log.info(f"[GPU] ✓ VA-API available on {render_device}")
    except Exception as e:
        log.info(f"[GPU] VA-API check failed: {e}")

    # Check QSV via ffmpeg
    qsv_ok = False
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-hwaccels"],
            capture_output=True, text=True, timeout=5)
        output = r.stdout + r.stderr
        qsv_ok = "qsv" in output.lower()
        if qsv_ok:
            log.info(f"[GPU] ✓ QSV available via FFmpeg")
    except Exception as e:
        log.info(f"[GPU] QSV check failed: {e}")

    return vaapi_ok, qsv_ok, render_device


_VAAPI_OK, _QSV_OK, _RENDER_DEV = _detect_gpu_decode()
log.info(f"[GPU] Decode backend: QSV={_QSV_OK}  VA-API={_VAAPI_OK}  device={_RENDER_DEV}")


# ─────────────────────────────────────────────────────────────────────────────
# NPU Inference Engine
# ─────────────────────────────────────────────────────────────────────────────
class NPUInferenceEngine:
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
        from openvino import Core
        core  = Core()
        avail = core.available_devices
        log.info(f"[OV] Available devices: {avail}")
        preferred = [d for d in ("NPU", "GPU", "CPU") if d in avail]
        dev = "AUTO:" + ",".join(preferred) if len(preferred) > 1 else (preferred[0] if preferred else "CPU")
        log.info(f"[NPU ENGINE] Compiling on {dev} …")
        config = {"PERFORMANCE_HINT": "THROUGHPUT", "CACHE_DIR": str(_SCRIPT_DIR / ".ov_cache")}
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
                config)
            self.compiled_models.append(cm)
            t = threading.Thread(target=self._worker, args=(i, cm), daemon=True, name=f"NPUWorker-{i}")
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
                if callback: callback(cam_id, frame, dets, inf_ms)
            except Exception as e:
                log.error(f"[NPUWorker-{wid}] cam {cam_id}: {e}")

    def get_stats(self):
        with self._lock:
            b = max(1, self.stats["batches"])
            return {"device": self.actual_dev, "batches": self.stats["batches"],
                    "avg_inf_ms": round(self.stats["inf_ms_sum"] / b, 1),
                    "dropped": self.stats["dropped"]}


# ─────────────────────────────────────────────────────────────────────────────
# NMS
# ─────────────────────────────────────────────────────────────────────────────
def _iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    u = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / u if u > 0 else 0.0

def _nms(dets, thr):
    if not dets: return []
    dets = sorted(dets, key=lambda x: x["conf"], reverse=True)
    keep = []; sup = [False] * len(dets)
    for i in range(len(dets)):
        if sup[i]: continue
        keep.append(dets[i])
        for j in range(i+1, len(dets)):
            if not sup[j] and _iou(dets[i]["bbox"], dets[j]["bbox"]) > thr:
                sup[j] = True
    return keep


# ─────────────────────────────────────────────────────────────────────────────
# Zone helpers
# ─────────────────────────────────────────────────────────────────────────────
def _in_poly(pt, poly):
    if not poly or len(poly) < 3: return False
    x,y = pt; inside=False; px,py = poly[-1]
    for nx,ny in poly:
        if ((ny>y)!=(py>y)) and x<(px-nx)*(y-ny)/(py-ny+1e-10)+nx:
            inside = not inside
        px,py = nx,ny
    return inside


# ─────────────────────────────────────────────────────────────────────────────
# Tripwire helpers
# ─────────────────────────────────────────────────────────────────────────────
def _cross_sign(px, py, ax, ay, bx, by) -> float:
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)

def _side(px, py, ax, ay, bx, by) -> int:
    c = _cross_sign(px, py, ax, ay, bx, by)
    return 1 if c > 0 else (-1 if c < 0 else 0)


# ─────────────────────────────────────────────────────────────────────────────
# Queue Analytics Engine — Zone-Change Debounce Method + Little's Law
# ─────────────────────────────────────────────────────────────────────────────
#
# COMPLETELY TRACKING-FREE. Uses only raw YOLO bounding boxes.
#
# HOW IT WORKS:
#
#   ZONE FLOW METHOD (every frame):
#     Count persons whose centroid is inside Queue Zone polygon → L_queue
#     Count persons whose centroid is inside Service Zone polygon → L_service
#     These update every inference frame from raw detections.
#
#   DEBOUNCE CROSSING DETECTION (zone count changes):
#     A count change must persist for DEBOUNCE consecutive frames before
#     it is confirmed as a real entry or exit. This prevents false crossings
#     from single-frame YOLO detection gaps.
#
#     Queue zone count DROPS by N  → N persons left queue  (_queue_exits)
#     Queue zone count RISES by N  → N persons entered queue (_queue_entries)
#     Service zone count DROPS by N → N persons left service (_service_exits)
#
#   LITTLE'S LAW (gives all 3 metrics):
#     Queue Length    = L_queue_now    (direct zone count)
#     Avg Wait Time   = L_queue_avg / λ_counter    (minutes → seconds)
#     Avg Proc Time   = L_service_avg / λ_exit     (minutes → seconds)
#     Throughput      = λ_exit × 60 persons/hour
#
# ─────────────────────────────────────────────────────────────────────────────
class QueueAnalytics:
    def __init__(self, alert_thr=10, wait_thr=300.0, cam_id=1):
        self.alert_thr = alert_thr
        self.wait_thr  = wait_thr
        self.cam_id    = cam_id

        # ── Zone polygons (normalised [0..1] coordinates) ─────────────────────
        self.zone_queue:   list = []
        self.zone_service: list = []
        self.zone_exit:    list = []

        # ── Tripwire lines (stored for dashboard display only) ────────────────
        self.counter_line: list = []
        self.exit_line:    list = []

        # ── Debounce crossing detector ─────────────────────────────────────────
        # A count change must persist for DEBOUNCE consecutive frames before
        # it is confirmed as a real entry or exit.  INIT_PHASE prevents false
        # crossings at startup before the baseline is set.
        self.DEBOUNCE   = 3
        self.INIT_PHASE = 5

        # Queue zone debounce state
        self._q_stable:      int = 0
        self._q_pend_val:    int = None
        self._q_pend_frames: int = 0
        self._q_init_frames: int = 0

        # Service zone debounce state
        self._s_stable:      int = 0
        self._s_pend_val:    int = None
        self._s_pend_frames: int = 0
        self._s_init_frames: int = 0

        # ── Rolling zone count samples for Little's Law (~5 min @ 1 fps) ──────
        self._L_queue_samples:   deque = deque(maxlen=300)
        self._L_service_samples: deque = deque(maxlen=300)

        # ── Flow counts ───────────────────────────────────────────────────────
        self._queue_entries: int = 0
        self._queue_exits:   int = 0
        self._service_exits: int = 0

        # ── Crossing event log for dashboard table ────────────────────────────
        self._cross_log: List[Dict] = []

        # ── Session timing ────────────────────────────────────────────────────
        self._t0           = time.time()
        self._frame_count  = 0

        # ── Consecutive empty-zone frame counters ─────────────────────────────
        # Once this exceeds _EMPTY_THRESH the time display is suppressed
        # (shown as "—") because the rolling-average L is stale.
        self._empty_q_frames = 0
        self._empty_s_frames = 0
        self._EMPTY_THRESH   = 10

        # ── Alerts ────────────────────────────────────────────────────────────
        self._ah = deque(maxlen=300)
        self.active_alerts: List[Dict] = []

        # ── Wait / proc time history — fallback before first crossing ─────────
        self._wait_history: deque = deque(maxlen=60)
        self._proc_history: deque = deque(maxlen=60)

        # ── Per-frame bbox→zone map for drawing color per bbox ────────────────
        self._tz: dict = {}

    # ── Configuration ─────────────────────────────────────────────────────────
    def set_zones(self, q, s, e):
        self.zone_queue   = q or []
        self.zone_service = s or []
        self.zone_exit    = e or []
        # Reset debounce state on zone change
        self._q_stable = 0; self._q_pend_val = None; self._q_pend_frames = 0; self._q_init_frames = 0
        self._s_stable = 0; self._s_pend_val = None; self._s_pend_frames = 0; self._s_init_frames = 0
        self._empty_q_frames = 0; self._empty_s_frames = 0

    def set_lines(self, counter_line: list, exit_line: list,
                  entry_dir: int = 1, exit_dir: int = -1):
        self.counter_line = counter_line or []
        self.exit_line    = exit_line    or []
        log.info(f"[Analytics cam {self.cam_id}] Lines set (display only — "
                 f"counting uses zone-change method)")

    def zones_defined(self) -> bool:
        return bool(self.zone_queue and self.zone_service)

    def lines_defined(self) -> bool:
        return bool(self.counter_line)

    def exit_zone_defined(self) -> bool:
        return bool(self.zone_exit) or bool(self.exit_line)

    # ── Multi-anchor zone check ─────────────────────────────────────────────
    # Dense 5×5 grid across the lower 2/3 of the bbox.  Checks 25 sample
    # points so that people near polygon edges are reliably detected.
    # Person is in-zone if ANY sample point hits the zone polygon.
    _ZONE_GRID_X = 5   # horizontal sample count
    _ZONE_GRID_Y = 5   # vertical sample count

    def _in_zone(self, bbox, fw, fh, zone_poly):
        """Return True if any sample point of bbox falls inside zone_poly."""
        if not zone_poly or len(zone_poly) < 3:
            return False

        x1n = bbox[0] / fw
        x2n = bbox[2] / fw
        top_y = bbox[1] / fh
        bot_y = bbox[3] / fh

        # Sample the lower 2/3 of the bbox (torso + legs — where the
        # person actually stands).  Skip the head region (top 1/3).
        body_top = top_y + (bot_y - top_y) * 0.33

        # Inset 10% from bbox edges horizontally
        bw = x2n - x1n
        margin = bw * 0.10
        lx = x1n + margin
        rx = x2n - margin
        if rx <= lx:        # very narrow bbox — just use center
            lx = rx = (x1n + x2n) / 2

        # Generate grid
        nx = self._ZONE_GRID_X
        ny = self._ZONE_GRID_Y
        for iy in range(ny):
            py = body_top + (bot_y - body_top) * iy / max(ny - 1, 1)
            for ix in range(nx):
                px = lx + (rx - lx) * ix / max(nx - 1, 1)
                if _in_poly((px, py), zone_poly):
                    return True
        return False

    # ── Main per-frame update — takes raw YOLO detections (NO TRACKING) ───────
    def update(self, dets, fw, fh):
        """
        dets: raw YOLO bounding boxes after NMS — list of
              {"bbox":[x1,y1,x2,y2], "conf":float, "cls":int}
              NO track_id needed.

        fw, fh: frame width and height in pixels
        """
        now = time.time()
        self._frame_count += 1
        tz = {}  # per-frame zone map — returned, NOT stored as shared state

        # ── STEP 1: Count persons in each zone this frame ─────────────────────
        L_queue_now   = 0
        L_service_now = 0

        for det in dets:
            bbox = det["bbox"]
            bbox_key = (int(bbox[0]), int(bbox[1]))
            # Priority: service > queue (handle overlap at boundary)
            if self._in_zone(bbox, fw, fh, self.zone_service):
                L_service_now += 1
                tz[bbox_key] = "service"
            elif self._in_zone(bbox, fw, fh, self.zone_queue):
                L_queue_now += 1
                tz[bbox_key] = "queue"

        # Track consecutive empty frames per zone
        if L_queue_now == 0:
            self._empty_q_frames += 1
        else:
            self._empty_q_frames = 0
        if L_service_now == 0:
            self._empty_s_frames += 1
        else:
            self._empty_s_frames = 0

        # ── STEP 2: Debounce crossing detection — Queue zone ──────────────────
        self._q_init_frames += 1
        if self._q_init_frames <= self.INIT_PHASE:
            self._q_stable = L_queue_now
            self._q_pend_val = L_queue_now
            self._q_pend_frames = self.DEBOUNCE
        else:
            if L_queue_now == self._q_pend_val:
                self._q_pend_frames += 1
            else:
                self._q_pend_val = L_queue_now
                self._q_pend_frames = 1

            if self._q_pend_frames >= self.DEBOUNCE:
                delta = self._q_pend_val - self._q_stable
                if delta < 0:
                    n = abs(delta)
                    self._queue_exits += n
                    for _ in range(n):
                        self._cross_log.append({
                            "time": time.strftime("%H:%M:%S"),
                            "type": "queue_exit",
                        })
                    if len(self._cross_log) > 50:
                        self._cross_log = self._cross_log[-50:]
                    log.info(f"[Cam {self.cam_id}] {n} queue exit(s)  "
                             f"total={self._queue_exits}")
                    self._q_stable = self._q_pend_val
                elif delta > 0:
                    self._queue_entries += delta
                    for _ in range(delta):
                        self._cross_log.append({
                            "time": time.strftime("%H:%M:%S"),
                            "type": "entry",
                        })
                    if len(self._cross_log) > 50:
                        self._cross_log = self._cross_log[-50:]
                    log.info(f"[Cam {self.cam_id}] {delta} queue entry(s)  "
                             f"total={self._queue_entries}")
                    self._q_stable = self._q_pend_val

        # ── STEP 3: Debounce crossing detection — Service zone ────────────────
        self._s_init_frames += 1
        if self._s_init_frames <= self.INIT_PHASE:
            self._s_stable = L_service_now
            self._s_pend_val = L_service_now
            self._s_pend_frames = self.DEBOUNCE
        else:
            if L_service_now == self._s_pend_val:
                self._s_pend_frames += 1
            else:
                self._s_pend_val = L_service_now
                self._s_pend_frames = 1

            if self._s_pend_frames >= self.DEBOUNCE:
                s_delta = self._s_pend_val - self._s_stable
                if s_delta < 0:
                    n = abs(s_delta)
                    self._service_exits += n
                    for _ in range(n):
                        self._cross_log.append({
                            "time": time.strftime("%H:%M:%S"),
                            "type": "service_exit",
                        })
                    if len(self._cross_log) > 50:
                        self._cross_log = self._cross_log[-50:]
                    log.info(f"[Cam {self.cam_id}] {n} service exit(s)  "
                             f"total={self._service_exits}")
                    self._s_stable = self._s_pend_val
                elif s_delta > 0:
                    self._s_stable = self._s_pend_val

        # ── STEP 4: Record occupancy samples for Little's Law ─────────────────
        self._L_queue_samples.append(L_queue_now)
        self._L_service_samples.append(L_service_now)

        # ── STEP 5: Little's Law calculations ─────────────────────────────────
        elapsed_min = max(0.017, (now - self._t0) / 60.0)

        L_queue   = (sum(self._L_queue_samples) / len(self._L_queue_samples)
                     if self._L_queue_samples else float(L_queue_now))
        L_service = (sum(self._L_service_samples) / len(self._L_service_samples)
                     if self._L_service_samples else float(L_service_now))

        lambda_exit    = self._queue_exits   / elapsed_min
        lambda_entry   = self._queue_entries / elapsed_min
        lambda_service = self._service_exits / elapsed_min

        if lambda_exit > 0:
            lambda_q = lambda_exit
        elif lambda_entry > 0:
            lambda_q = lambda_entry
        else:
            lambda_q = 0.0

        # ── Warmup guard ──────────────────────────────────────────────────────
        # Brief warmup to let the rate stabilise before displaying values.
        _WARMUP_MIN   = 0.25    # 15 seconds
        _MIN_EXITS    = 1      # at least 1 observed exit
        warmed_up_q   = elapsed_min >= _WARMUP_MIN and self._queue_exits >= _MIN_EXITS
        warmed_up_s   = elapsed_min >= _WARMUP_MIN and self._service_exits >= _MIN_EXITS

        # Avg Waiting Time = L_queue / λ_q  [Little's Law]
        if lambda_q > 0 and warmed_up_q:
            avg_wait_s  = (L_queue / lambda_q) * 60.0
            wait_method = "littles_law"
            self._wait_history.append(avg_wait_s)
        elif lambda_q > 0 and not warmed_up_q:
            avg_wait_s  = 0.0
            wait_method = "warming_up"
        elif L_queue_now > 0 and self._wait_history:
            avg_wait_s  = sum(self._wait_history) / len(self._wait_history)
            wait_method = "estimated"
        else:
            avg_wait_s  = 0.0
            wait_method = "no_data" if L_queue_now > 0 else "none"

        if self._empty_q_frames >= self._EMPTY_THRESH:
            avg_wait_s  = 0.0
            wait_method = "none"

        # Avg Processing Time = L_service / λ_service  [Little's Law]
        if lambda_service > 0 and warmed_up_s:
            avg_proc_s  = (L_service / lambda_service) * 60.0
            proc_method = "littles_law"
            self._proc_history.append(avg_proc_s)
        elif lambda_service > 0 and not warmed_up_s:
            avg_proc_s  = 0.0
            proc_method = "warming_up"
        elif L_service_now > 0 and self._proc_history:
            avg_proc_s  = sum(self._proc_history) / len(self._proc_history)
            proc_method = "estimated"
        else:
            avg_proc_s  = 0.0
            proc_method = "no_data" if L_service_now > 0 else "none"

        if self._empty_s_frames >= self._EMPTY_THRESH:
            avg_proc_s  = 0.0
            proc_method = "none"

        throughput_hr = lambda_service * 60.0

        open_counter = (
            avg_wait_s > self.wait_thr
            and L_queue_now > 0
            and lambda_exit == 0
        )

        # ── STEP 6: Alerts ────────────────────────────────────────────────────
        new_a = []

        def _alert(tp, msg, lv):
            last = [a.get("_ts", 0) for a in self._ah
                    if a.get("type") == tp and not a.get("acked")]
            if not last or now - last[-1] > 30:
                idx = len(self._ah)
                a = {"idx": idx, "time": time.strftime("%H:%M:%S"),
                     "message": msg, "level": lv, "type": tp,
                     "_ts": now, "queue_length": L_queue_now,
                     "avg_wait": round(avg_wait_s, 1), "acked": False}
                self._ah.append(a); new_a.append(a)
                self._persist_alert(a)

        if L_queue_now > self.alert_thr:
            _alert("queue_length",
                   f"Queue {L_queue_now} > threshold ({self.alert_thr})",
                   "critical" if L_queue_now > self.alert_thr * 1.5 else "warning")
        if avg_wait_s > self.wait_thr and wait_method not in ("none", "no_data"):
            _alert("wait_time",
                   f"Avg wait {avg_wait_s:.0f}s > {self.wait_thr:.0f}s", "warning")
        if open_counter:
            _alert("open_counter",
                   f"No exits detected — consider opening a new counter "
                   f"(Q={L_queue_now}, wait={avg_wait_s:.0f}s)",
                   "critical")
        self.active_alerts = new_a

        return {
            "queue_length":          L_queue_now,
            "L_queue_avg":           round(L_queue,   1),
            "L_service_avg":         round(L_service, 1),
            "avg_waiting_time":      round(avg_wait_s, 1),
            "open_counter":          open_counter,
            "avg_processing_time":   round(avg_proc_s, 1),
            "wait_method":           wait_method,
            "proc_method":           proc_method,
            "queue_empty_frames":    self._empty_q_frames,
            "queue_entries":         self._queue_entries,
            "queue_exits":           self._queue_exits,
            "service_exits":         self._service_exits,
            "lambda_entry_pm":       round(lambda_entry,   3),
            "lambda_exit_pm":        round(lambda_exit,    3),
            "lambda_q_pm":           round(lambda_q,       3),
            "lambda_service_pm":     round(lambda_service, 3),
            # Backward-compat keys (dashboard still reads these)
            "counter_crossings":     self._queue_exits,
            "exit_crossings":        self._service_exits,
            "entry_crossings":       self._queue_entries,
            "lambda_counter_pm":     round(lambda_q,       3),
            "throughput_per_hour":   round(throughput_hr,  1),
            "total_processed":       self._service_exits,
            "zone_counts": {
                "queue":   L_queue_now,
                "service": L_service_now,
                "exit":    0,
            },
            "crossing_log":          self._cross_log[-10:],
            "alerts":                new_a,
            # Legacy fields
            "max_waiting_time":      0.0,
            "max_processing_time":   0.0,
            "tput_method":           "observed" if self._service_exits > 0 else "none",
            "seeded_count":          0,
            "partial_wait_samples":  0,
            "queue_dwells":          {},
            "service_dwells":        {},
            "warnings":              [],
            "_tz":                   tz,
        }

    def get_alert_history(self) -> List[Dict]:
        return [{k: v for k, v in a.items() if k != "_ts"} for a in self._ah]

    def acknowledge_alert(self, idx: int) -> bool:
        for a in self._ah:
            if a.get("idx") == idx:
                a["acked"] = True; self._rewrite_alerts_file(); return True
        return False

    def get_counter_performance(self) -> Dict:
        elapsed_min  = max(0.017, (time.time() - self._t0) / 60.0)
        lambda_svc   = self._service_exits / elapsed_min
        lambda_q_exit = self._queue_exits  / elapsed_min
        L_service    = (sum(self._L_service_samples) / len(self._L_service_samples)
                        if self._L_service_samples else 0.0)
        avg_proc     = (L_service / lambda_svc * 60.0) if lambda_svc > 0 else None
        return {
            "samples":          self._service_exits,
            "min_s": None, "max_s": None,
            "avg_s":            round(avg_proc, 1) if avg_proc else None,
            "p50_s": None, "p90_s": None,
            "total_processed":  self._service_exits,
            "queue_entries":    self._queue_entries,
            "queue_exits":      self._queue_exits,
            "lambda_exit_pm":   round(lambda_q_exit, 3),
            "lambda_service_pm":round(lambda_svc,    3),
            "L_service_avg":    round(L_service, 2),
        }

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
            with open(path, "w") as f: json.dump(existing, f, indent=2)
        except Exception as e:
            log.warning(f"[Analytics cam {self.cam_id}] alert persist: {e}")

    def _rewrite_alerts_file(self):
        try:
            path = _ALERTS_DIR / f"cam_{self.cam_id}_alerts.json"
            data = [{k: v for k, v in a.items() if k != "_ts"} for a in self._ah]
            with open(path, "w") as f: json.dump(data, f, indent=2)
        except Exception as e:
            log.warning(f"[Analytics cam {self.cam_id}] alerts rewrite: {e}")

    def reset(self):
        self._q_stable = 0; self._q_pend_val = None; self._q_pend_frames = 0; self._q_init_frames = 0
        self._s_stable = 0; self._s_pend_val = None; self._s_pend_frames = 0; self._s_init_frames = 0
        self._L_queue_samples.clear(); self._L_service_samples.clear()
        self._queue_entries = 0; self._queue_exits = 0; self._service_exits = 0
        self._cross_log.clear(); self._tz.clear()
        self._t0 = time.time(); self._frame_count = 0; self._ah.clear()
        self._wait_history.clear(); self._proc_history.clear()
        self._empty_q_frames = 0; self._empty_s_frames = 0

    def smooth_reset(self):
        """
        Called at every video loop boundary — treats the looping file as a
        continuous stream by preserving all cumulative counters and elapsed
        time, but flushing only the debounce pending state.

        What is preserved (continuous stream semantics):
          _queue_entries, _queue_exits, _service_exits  — keep accumulating
          _q_stable, _s_stable                          — known pre-loop baseline
          _t0                                           — session clock never resets
          _wait_history, _proc_history                  — fallback pools intact
          _L_queue_samples, _L_service_samples          — rolling occupancy history
          _ah (alert history)                           — alerts carry over

        What is flushed (debounce pending state only):
          _q_pend_val/frames, _s_pend_val/frames — pending tracker reset
          _tz                                    — per-frame bbox→zone map
        """
        self._q_pend_val = None; self._q_pend_frames = 0
        self._s_pend_val = None; self._s_pend_frames = 0
        self._tz            = {}
        self._empty_q_frames = 0; self._empty_s_frames = 0
        log.info(f"[Analytics cam {self.cam_id}] smooth_reset — "
                 f"loop boundary, counters preserved "
                 f"(exits={self._queue_exits} entries={self._queue_entries} "
                 f"svc_exits={self._service_exits} "
                 f"t0_age={(time.time()-self._t0):.0f}s)")


# ─────────────────────────────────────────────────────────────────────────────
# GPU Decode helpers — GStreamer pipeline builder
# ─────────────────────────────────────────────────────────────────────────────
def _check_gstreamer() -> bool:
    """Check if OpenCV was built with GStreamer support."""
    try:
        build = cv2.getBuildInformation()
        return "GStreamer" in build and "YES" in build[build.find("GStreamer"):build.find("GStreamer")+40]
    except:
        return False

def _check_vaapi_gst() -> bool:
    """Check if GStreamer VA-API plugin (vaapi or va) is available."""
    try:
        r = subprocess.run(
            ["gst-inspect-1.0", "vaapidecodebin"],
            capture_output=True, timeout=4)
        if r.returncode == 0: return True
        r = subprocess.run(
            ["gst-inspect-1.0", "vah264dec"],
            capture_output=True, timeout=4)
        return r.returncode == 0
    except:
        return False

def _check_qsv_gst() -> bool:
    """Check if GStreamer Intel QSV plugin is available."""
    try:
        r = subprocess.run(
            ["gst-inspect-1.0", "msdkh264dec"],
            capture_output=True, timeout=4)
        return r.returncode == 0
    except:
        return False

_GST_AVAILABLE   = _check_gstreamer()
_GST_VAAPI_OK    = _check_vaapi_gst()   if _GST_AVAILABLE else False
_GST_QSV_OK      = _check_qsv_gst()    if _GST_AVAILABLE else False

log.info(f"[GPU] GStreamer: {_GST_AVAILABLE}  GST-VA-API: {_GST_VAAPI_OK}  GST-QSV: {_GST_QSV_OK}")


def _build_gst_pipeline(src: str, is_stream: bool, render_dev: str) -> list:
    """
    Build GStreamer pipeline strings ordered by reliability for Intel iGPU.

    Returns list of (pipeline_string, label) tuples to try in order.
    """
    pipelines = []

    if is_stream:
        src_h264 = (f'rtspsrc location="{src}" latency=50 '
                    f'! rtph264depay ! h264parse')
        src_h265 = (f'rtspsrc location="{src}" latency=50 '
                    f'! rtph265depay ! h265parse')
    else:
        src_h264 = f'filesrc location="{src}" ! qtdemux name=d d.video_0 ! h264parse'
        src_h265 = f'filesrc location="{src}" ! qtdemux name=d d.video_0 ! h265parse'

    if _GST_VAAPI_OK:
        pipelines.append((
            f'{src_h264} ! vaapidecodebin ! vaapipostproc ! '
            f'videoconvert ! video/x-raw,format=BGR ! appsink drop=1 sync=0',
            "GST-VAAPI-H264"))

    if _GST_VAAPI_OK and not is_stream:
        pipelines.append((
            f'{src_h265} ! vaapidecodebin ! vaapipostproc ! '
            f'videoconvert ! video/x-raw,format=BGR ! appsink drop=1 sync=0',
            "GST-VAAPI-H265"))

    if _GST_VAAPI_OK:
        pipelines.append((
            f'{src_h264} ! vaapidecodebin ! '
            f'videoconvert ! video/x-raw,format=BGR ! appsink drop=1 sync=0',
            "GST-VAAPI-DIRECT"))

    if _GST_VAAPI_OK:
        pipelines.append((
            f'{src_h264} ! vah264dec ! '
            f'videoconvert ! video/x-raw,format=BGR ! appsink drop=1 sync=0',
            "GST-VA-H264"))

    if _GST_VAAPI_OK and not is_stream:
        pipelines.append((
            f'filesrc location="{src}" ! decodebin ! '
            f'vaapipostproc ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1 sync=0',
            "GST-VAAPI-DECODEBIN"))

    if _GST_QSV_OK:
        pipelines.append((
            f'{src_h264} ! msdkh264dec ! '
            f'videoconvert ! video/x-raw,format=BGR ! appsink drop=1 sync=0',
            "GST-QSV-H264"))

    if _GST_AVAILABLE and not is_stream:
        pipelines.append((
            f'filesrc location="{src}" ! decodebin ! '
            f'videoconvert ! video/x-raw,format=BGR ! appsink drop=1 sync=0',
            "GST-AUTO"))

    return pipelines


# ─────────────────────────────────────────────────────────────────────────────
# Video Decoder  — GStreamer GPU decode → FFmpeg VA-API → CPU fallback
# ─────────────────────────────────────────────────────────────────────────────
class VideoDecoder:
    """
    Hardware decode priority:
      1. GStreamer + vaapidecodebin  (VA-API, Intel iGPU)
      2. GStreamer + vah264dec       (newer VA plugin)
      3. GStreamer + msdkh264dec     (Intel QSV/MSDK)
      4. GStreamer + decodebin       (auto-selects)
      5. FFmpeg VA-API
      6. FFmpeg plain / CPU fallback
    """
    def __init__(self):
        self.cap           = None
        self._frame        = None
        self._lock         = threading.Lock()
        self._thread       = None
        self._run          = False
        self.source        = None
        self.mode          = "idle"
        self.fps           = 0.0
        self._fc           = 0
        self._t0           = 0.0
        self.decode_method = "cpu"
        self.on_loop_restart: Optional[Callable] = None

    def start(self, source: str, mode: str = "live"):
        self.stop()
        self.source = source; self.mode = mode
        self._run = True; self._fc = 0; self._t0 = time.time()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name=f"Decode-{str(source)[:30]}")
        self._thread.start()
        log.info(f"[Decoder] {mode}: {source}")

    def stop(self):
        self._run = False
        if self._thread: self._thread.join(timeout=3)
        if self.cap: self.cap.release(); self.cap = None
        self._frame = None; self.mode = "idle"

    def _try_gst(self, pipeline: str, label: str):
        try:
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    log.info(f"[Decoder] ✓ {label}  "
                             f"({frame.shape[1]}×{frame.shape[0]})")
                    return cap
                cap.release()
        except Exception as e:
            log.debug(f"[Decoder] {label} failed: {e}")
        return None

    def _try_ffmpeg(self, src, options: str, label: str):
        try:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = options
            cap = cv2.VideoCapture(str(src), cv2.CAP_FFMPEG)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    log.info(f"[Decoder] ✓ {label}: {str(src)[:45]}")
                    return cap
                cap.release()
        except Exception as e:
            log.debug(f"[Decoder] {label} failed: {e}")
        finally:
            os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
        return None

    def _open_best(self, src) -> Tuple[Optional[object], str]:
        src_str = str(src)
        is_stream = src_str.startswith("rtsp") or \
                    src_str.startswith("http") or \
                    src_str.startswith("rtp")

        # ── 1. GStreamer GPU pipelines ─────────────────────────────────────────
        if _GST_AVAILABLE:
            for pipeline, label in _build_gst_pipeline(src_str, is_stream, _RENDER_DEV):
                cap = self._try_gst(pipeline, label)
                if cap:
                    return cap, label.lower().replace("-","_")

        # ── 2. FFmpeg + VA-API ────────────────────────────────────────────────
        if _VAAPI_OK:
            cap = self._try_ffmpeg(src,
                f"video_codec;h264_vaapi|hwaccel;vaapi|hwaccel_device;{_RENDER_DEV}|"
                f"hwaccel_output_format;nv12",
                "FFmpeg-VAAPI-H264")
            if cap: return cap, "ffmpeg_vaapi"

        # ── 3. FFmpeg + QSV ───────────────────────────────────────────────────
        if _QSV_OK:
            cap = self._try_ffmpeg(src,
                f"video_codec;h264_qsv|hwaccel;qsv|hwaccel_device;{_RENDER_DEV}",
                "FFmpeg-QSV-H264")
            if cap: return cap, "ffmpeg_qsv"

        # ── 4. FFmpeg plain (for RTSP streams) ────────────────────────────────
        if is_stream:
            try:
                cap = cv2.VideoCapture(src_str, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if cap.isOpened():
                    log.info(f"[Decoder] ✓ FFmpeg CPU (stream): {src_str[:45]}")
                    return cap, "cpu_ffmpeg"
                cap.release()
            except: pass

        # ── 5. Plain CPU fallback ─────────────────────────────────────────────
        try:
            cap = cv2.VideoCapture(src_str)
            if cap.isOpened():
                log.warning(f"[Decoder] ⚠ CPU software decode: {src_str[:45]}")
                return cap, "cpu"
            cap.release()
        except: pass

        return None, "none"

    def _loop(self):
        src = self.source
        try: src = int(src)
        except: pass

        def _is_local_file(s):
            if isinstance(s, int): return False
            return not (str(s).startswith("rtsp") or
                        str(s).startswith("http") or
                        str(s).startswith("rtp"))

        if _is_local_file(src) and self.mode != "recording":
            log.info(f"[Decoder] Local file detected — switching to recording mode")
            self.mode = "recording"

        def _open():
            cap, method = self._open_best(src)
            if cap is None:
                log.error(f"[Decoder] Cannot open: {self.source}")
            else:
                self.decode_method = method
                log.info(f"[Decoder] Active decode: {method}")
            return cap

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
                    loop_count += 1
                    log.info(f"[Decoder] '{str(self.source)[:40]}' "
                             f"end of file — restarting loop #{loop_count}")
                    cap.release()
                    time.sleep(0.15)
                    cap = _open()
                    if cap is None:
                        log.error("[Decoder] Cannot reopen — stopping")
                        self._run = False; break
                    self.cap = cap
                    native_fps = cap.get(cv2.CAP_PROP_FPS)
                    if native_fps <= 0 or native_fps > 120: native_fps = 25.0
                    frame_interval = 1.0 / native_fps
                    if self.on_loop_restart:
                        try: self.on_loop_restart(loop_count)
                        except Exception as e:
                            log.warning(f"[Decoder] on_loop_restart error: {e}")
                    continue
                else:
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
ZONE_COLORS = {"queue":(0,200,255), "service":(0,255,128), "exit":(200,80,255)}
PERSON_C    = {"queue":(0,200,255), "service":(0,255,128),
               "exit":(200,80,255), None:(180,180,180)}


class CameraStream:
    _id_counter = 1

    def __init__(self, name: str, source: str, mode: str="live",
                 alert_thr: int=10, wait_thr: float=300.0, inf_fps: float=15.0):
        self.cam_id  = CameraStream._id_counter; CameraStream._id_counter += 1
        self.name    = name
        self.source  = source
        self.mode    = mode
        self.inf_fps_limit: float = max(1.0, min(30.0, float(inf_fps)))
        self.decoder   = VideoDecoder()
        # Analytics takes raw YOLO detections directly — no tracking layer
        self.analytics = QueueAnalytics(alert_thr, wait_thr, cam_id=self.cam_id)
        self._lock     = threading.Lock()
        self._metrics: Dict = {}
        self._ann:     Optional[np.ndarray] = None
        self._inf_fps: float = 0.0
        self._cb_fc:   int   = 0
        self._cb_t:    float = time.time()
        self._inf_ms:  float = 0.0
        self._history: deque = deque(maxlen=3600)
        self.active:           bool = False
        self.inference_active: bool = False
        self._last_sub: float = 0.0

    def start(self, npu_engine: "NPUInferenceEngine"):
        self.decoder.start(self.source, self.mode)
        def _on_loop(loop_count):
            self.analytics.smooth_reset()
        self.decoder.on_loop_restart = _on_loop
        self._npu   = npu_engine
        self.active = True
        t = threading.Thread(target=self._submit_loop, daemon=True,
                             name=f"Cam-{self.cam_id}")
        t.start()
        log.info(f"[Cam-{self.cam_id}] '{self.name}' started (zone-change debounce crossings)")

    def stop(self):
        self.active = False; self.decoder.stop()
        log.info(f"[Cam-{self.cam_id}] stopped")

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

    def _on_inference(self, cam_id: int, frame: np.ndarray, raw_dets: list, inf_ms: float):
        h, w = frame.shape[:2]

        # Raw YOLO detections go directly to QueueAnalytics — no tracking
        if self.analytics.zones_defined():
            metrics = self.analytics.update(raw_dets, w, h)
        else:
            metrics = {
                "queue_length":0, "avg_waiting_time":0, "avg_processing_time":0,
                "throughput_per_hour":0, "tput_method":"none", "total_processed":0,
                "counter_crossings":0, "exit_crossings":0, "entry_crossings":0,
                "lambda_counter_pm":0, "lambda_exit_pm":0,
                "L_queue_avg":0, "L_service_avg":0,
                "wait_method":"none", "proc_method":"none",
                "open_counter":False,
                "alerts":[], "zone_counts":{}, "crossing_log":[],
                "warnings":["Draw zones in Zone Setup tab"],
                "queue_dwells":{},"service_dwells":{},"seeded_count":0,
                "partial_wait_samples":0, "max_waiting_time":0, "max_processing_time":0,
                "_tz": {},
            }

        tz = metrics.pop("_tz", {})
        ann = self._draw(frame.copy(), raw_dets, metrics, inf_ms, tz)
        self._cb_fc += 1
        now = time.time()
        if now - self._cb_t >= 1.0:
            self._inf_fps = self._cb_fc / (now - self._cb_t)
            self._cb_fc = 0; self._cb_t = now
        self._inf_ms = inf_ms
        with self._lock:
            self._metrics = metrics
            self._ann     = ann
            last_ts = self._history[-1].get("_ts",0) if self._history else 0
            if now - last_ts >= 1.0:
                entry = {
                    "ts":              time.strftime("%H:%M:%S"),
                    "queue_length":    metrics["queue_length"],
                    "avg_wait":        metrics["avg_waiting_time"],
                    "wait_method":     metrics.get("wait_method","none"),
                    "avg_proc":        metrics["avg_processing_time"],
                    "throughput":      metrics["throughput_per_hour"],
                    "in_service":      metrics.get("zone_counts",{}).get("service",0),
                    "total_processed": metrics.get("total_processed",0),
                    "queue_entries":   metrics.get("queue_entries",0),
                    "queue_exits":     metrics.get("queue_exits",0),
                    "open_counter":    metrics.get("open_counter",False),
                    "_ts":             now,
                }
                self._history.append(entry)
                if len(self._history) % 60 == 0:
                    self._persist_history()

    def _persist_history(self):
        try:
            path = _ALERTS_DIR / f"cam_{self.cam_id}_history.json"
            data = [{k:v for k,v in e.items() if k!="_ts"} for e in self._history]
            with open(path,"w") as f: json.dump(data, f)
            log.info(f"[Cam-{self.cam_id}] History saved ({len(data)} samples)")
        except Exception as e:
            log.warning(f"[Cam-{self.cam_id}] history persist failed: {e}")

    # def _draw(self, frame, dets, metrics, inf_ms, tz=None) -> np.ndarray:
    #     """
    #     Clean minimal overlay:
    #     • Bounding boxes colored by zone (Q=cyan, S=green, unzoned=grey)
    #     • Compact HUD strip at top — does NOT cover the video
    #     """
    #     h, w = frame.shape[:2]
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     if tz is None:
    #         tz = {}

    #     zone_colors = {
    #         "queue":   (0, 200, 255),   # cyan  — person in queue zone
    #         "service": (0, 230, 100),   # green — person in service zone
    #     }

    #     # ── Draw zone polygons on the frame so the user sees exactly
    #     #    what the backend considers to be "in zone" ─────────────────────
    #     for zone_name, poly in [("queue",   self.analytics.zone_queue),
    #                             ("service", self.analytics.zone_service)]:
    #         if poly and len(poly) >= 3:
    #             pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in poly], np.int32)
    #             overlay = frame.copy()
    #             cv2.fillPoly(overlay, [pts], zone_colors[zone_name])
    #             cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    #             cv2.polylines(frame, [pts], True, zone_colors[zone_name], 2, cv2.LINE_AA)
    #             # Label
    #             cx = int(sum(p[0] for p in poly) / len(poly) * w)
    #             cy = int(sum(p[1] for p in poly) / len(poly) * h)
    #             cv2.putText(frame, zone_name.upper(), (cx - 20, cy),
    #                         font, 0.5, zone_colors[zone_name], 1, cv2.LINE_AA)

    #     for det in dets:
    #         x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
    #         bbox_key = (x1, y1)
    #         zone = tz.get(bbox_key)
    #         if zone == "queue":
    #             cv2.rectangle(frame, (x1,y1), (x2,y2), zone_colors["queue"], 2)
    #             cv2.putText(frame, "Q", (x1+3, y1+14),
    #                         font, 0.45, zone_colors["queue"], 1, cv2.LINE_AA)
    #         elif zone == "service":
    #             cv2.rectangle(frame, (x1,y1), (x2,y2), zone_colors["service"], 2)
    #             cv2.putText(frame, "S", (x1+3, y1+14),
    #                         font, 0.45, zone_colors["service"], 1, cv2.LINE_AA)
    #         else:
    #             cv2.rectangle(frame, (x1,y1), (x2,y2), (90,90,90), 1)

    #     # ── Compact HUD ───────────────────────────────────────────────────────
    #     wm   = metrics.get("wait_method","none")
    #     wlbl = ("L" if wm=="littles_law" else
    #             "~" if wm=="estimated" else
    #             "W" if wm=="warming_up" else "—")
    #     pm   = metrics.get("proc_method","none")
    #     plbl = ("L" if pm=="littles_law" else
    #             "~" if pm=="estimated" else
    #             "W" if pm=="warming_up" else "—")
    #     ql   = metrics.get("queue_length", 0)
    #     sl   = metrics.get("zone_counts",{}).get("service", 0)
    #     awt  = metrics.get("avg_waiting_time", 0.0)
    #     apt  = metrics.get("avg_processing_time", 0.0)
    #     qin  = metrics.get("queue_entries", 0)
    #     qout = metrics.get("queue_exits",   0)
    #     scx  = metrics.get("service_exits", 0)
    #     tph  = metrics.get("throughput_per_hour", 0.0)
    #     dm   = self.decoder.decode_method.upper()
    #     oc   = "⚠ OPEN COUNTER" if metrics.get("open_counter") else ""

    #     awt_str = f"{awt:.0f}s" if wm not in ("none","no_data","warming_up") else "—"
    #     apt_str = f"{apt:.0f}s" if pm not in ("none","no_data","warming_up") else "—"
    #     hud_text = (
    #         f"Q:{ql} S:{sl} | "
    #         f"Wait[{wlbl}]:{awt_str}  Proc[{plbl}]:{apt_str} | "
    #         f"In:{qin} Out:{qout} Tput:{tph:.0f}/h | "
    #         f"{self._inf_fps:.0f}fps {inf_ms:.0f}ms"
    #         + (f" | {oc}" if oc else "")
    #     )

    #     (tw, th), _ = cv2.getTextSize(hud_text, font, 0.42, 1)
    #     strip_h = th + 10

    #     ov = frame.copy()
    #     cv2.rectangle(ov, (0, 0), (w, strip_h + 4), (10, 10, 10), -1)
    #     cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
    #     cv2.putText(frame, hud_text, (6, strip_h),
    #                 font, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

    #     # ── Alert banner ──────────────────────────────────────────────────────
    #     if metrics.get("alerts"):
    #         al  = metrics["alerts"][0]
    #         msg = al["message"]
    #         bg  = (0,0,160) if al.get("level") == "critical" else (0,60,120)
    #         (mw, mh), _ = cv2.getTextSize(msg, font, 0.45, 1)
    #         bx = (w - mw) // 2
    #         by = h - 12
    #         cv2.rectangle(frame, (bx-8, by-mh-6), (bx+mw+8, by+4), bg, -1)
    #         cv2.putText(frame, msg, (bx, by), font, 0.45,
    #                     (255, 220, 120), 1, cv2.LINE_AA)

    #     return frame

    def _draw(self, frame, dets, metrics, inf_ms, tz=None) -> np.ndarray:
        """
        Resolution-aware overlay:
        • Bounding boxes colored by zone (Q=cyan, S=green, unzoned=grey)
        • Compact HUD strip at top — does NOT cover the video
        All line thicknesses, font sizes, and opacities scale with the
        frame's diagonal so the overlay looks clean at any resolution.
        """
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        if tz is None:
            tz = {}
 
        # ── Resolution-aware scale ────────────────────────────────────────────
        # Normalised so that a 640×480 frame gives scale ≈ 1.0
        diag        = (w * w + h * h) ** 0.5          # e.g. 800 for 640×480
        draw_scale  = diag / 800.0                    # 1.0 @ 640×480
                                                      # 1.84 @ 1280×720
                                                      # 2.75 @ 1920×1080
 
        # Line thicknesses (always at least 1 px)
        box_thick   = max(1, round(draw_scale))        # det bounding boxes
        zone_thick  = max(1, round(draw_scale * 1.5))  # zone polygon borders
 
        # Font scales
        lbl_scale   = max(0.25, 0.38 * draw_scale)    # box "Q"/"S" labels
        hud_scale   = max(0.32, 0.42 * draw_scale)    # HUD strip text
 
        # Zone fill — lighter on high-res frames to avoid swamping the image
        fill_alpha  = max(0.05, 0.15 / draw_scale)    # ~0.15 @ 640p, ~0.05 @ 1080p
 
        # Minimum box height (pixels) below which we skip the "Q"/"S" label
        min_lbl_h   = max(14, int(18 * draw_scale))
 
        zone_colors = {
            "queue":   (0, 200, 255),   # cyan  — person in queue zone
            "service": (0, 230, 100),   # green — person in service zone
        }
 
        # ── Draw zone polygons ────────────────────────────────────────────────
        for zone_name, poly in [("queue",   self.analytics.zone_queue),
                                 ("service", self.analytics.zone_service)]:
            if poly and len(poly) >= 3:
                pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in poly],
                                np.int32)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], zone_colors[zone_name])
                cv2.addWeighted(overlay, fill_alpha, frame, 1.0 - fill_alpha,
                                0, frame)
                cv2.polylines(frame, [pts], True, zone_colors[zone_name],
                              zone_thick, cv2.LINE_AA)
                # Zone label — centroid
                cx = int(sum(p[0] for p in poly) / len(poly) * w)
                cy = int(sum(p[1] for p in poly) / len(poly) * h)
                cv2.putText(frame, zone_name.upper(), (cx - 20, cy),
                            font, lbl_scale * 1.1,
                            zone_colors[zone_name], box_thick, cv2.LINE_AA)
 
        # ── Draw detection bounding boxes ─────────────────────────────────────
        for det in dets:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            bbox_key = (x1, y1)
            zone     = tz.get(bbox_key)
            box_h    = y2 - y1
 
            if zone == "queue":
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              zone_colors["queue"], box_thick)
                if box_h >= min_lbl_h:
                    cv2.putText(frame, "Q", (x1 + 3, y1 + int(13 * draw_scale)),
                                font, lbl_scale,
                                zone_colors["queue"], box_thick, cv2.LINE_AA)
 
            elif zone == "service":
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              zone_colors["service"], box_thick)
                if box_h >= min_lbl_h:
                    cv2.putText(frame, "S", (x1 + 3, y1 + int(13 * draw_scale)),
                                font, lbl_scale,
                                zone_colors["service"], box_thick, cv2.LINE_AA)
 
            else:
                # Un-zoned person — thin grey box, no label
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (80, 80, 80), max(1, box_thick - 1))
 
        # ── Compact HUD ───────────────────────────────────────────────────────
        wm   = metrics.get("wait_method", "none")
        wlbl = ("L" if wm == "littles_law" else
                "~" if wm == "estimated"   else
                "W" if wm == "warming_up"  else "—")
        pm   = metrics.get("proc_method", "none")
        plbl = ("L" if pm == "littles_law" else
                "~" if pm == "estimated"   else
                "W" if pm == "warming_up"  else "—")
        ql   = metrics.get("queue_length", 0)
        sl   = metrics.get("zone_counts", {}).get("service", 0)
        awt  = metrics.get("avg_waiting_time", 0.0)
        apt  = metrics.get("avg_processing_time", 0.0)
        qin  = metrics.get("queue_entries", 0)
        qout = metrics.get("queue_exits", 0)
        tph  = metrics.get("throughput_per_hour", 0.0)
        oc   = "⚠ OPEN COUNTER" if metrics.get("open_counter") else ""
 
        awt_str = (f"{awt:.0f}s"
                   if wm not in ("none", "no_data", "warming_up") else "—")
        apt_str = (f"{apt:.0f}s"
                   if pm not in ("none", "no_data", "warming_up") else "—")
        hud_text = (
            f"Q:{ql} S:{sl} | "
            f"Wait[{wlbl}]:{awt_str}  Proc[{plbl}]:{apt_str} | "
            f"In:{qin} Out:{qout} Tput:{tph:.0f}/h | "
            f"{self._inf_fps:.0f}fps {inf_ms:.0f}ms"
            + (f" | {oc}" if oc else "")
        )
 
        (tw, th), _ = cv2.getTextSize(hud_text, font, hud_scale, 1)
        strip_h     = th + max(8, int(10 * draw_scale))
 
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, strip_h + 4), (10, 10, 10), -1)
        cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, hud_text, (6, strip_h),
                    font, hud_scale, (220, 220, 220), 1, cv2.LINE_AA)
 
        # ── Alert banner ──────────────────────────────────────────────────────
        if metrics.get("alerts"):
            al  = metrics["alerts"][0]
            msg = al["message"]
            bg  = (0, 0, 160) if al.get("level") == "critical" else (0, 60, 120)
            (mw, mh), _ = cv2.getTextSize(msg, font, hud_scale, 1)
            bx = (w - mw) // 2
            by = h - 12
            cv2.rectangle(frame,
                          (bx - 8, by - mh - 6), (bx + mw + 8, by + 4),
                          bg, -1)
            cv2.putText(frame, msg, (bx, by),
                        font, hud_scale, (255, 220, 120), 1, cv2.LINE_AA)
 
        return frame

    def get_metrics(self) -> Dict:
        with self._lock: return dict(self._metrics)

    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            if self._ann is not None:
                ret,buf = cv2.imencode(".jpg",self._ann,[cv2.IMWRITE_JPEG_QUALITY,80])
                return buf.tobytes() if ret else None
        raw = self.decoder.read()
        if raw is None: return None
        ret,buf = cv2.imencode(".jpg",raw,[cv2.IMWRITE_JPEG_QUALITY,75])
        return buf.tobytes() if ret else None

    def get_history(self) -> List:
        with self._lock:
            return [{k:v for k,v in e.items() if k!="_ts"} for e in self._history]

    def get_history_csv(self) -> str:
        with self._lock:
            rows = [{k:v for k,v in e.items() if k!="_ts"} for e in self._history]
        if not rows:
            return "ts,queue_length,avg_wait,avg_proc,throughput,in_service,total_processed\n"
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
        return buf.getvalue()

    def set_zones(self, q, s, e):
        with self._lock:
            self.analytics.set_zones(q, s, e)

    def set_lines(self, entry_line, exit_line, entry_dir=1, exit_dir=-1):
        with self._lock:
            self.analytics.set_lines(entry_line, exit_line, entry_dir, exit_dir)

    def set_thresholds(self, alert_thr, wait_thr):
        with self._lock:
            self.analytics.alert_thr = alert_thr
            self.analytics.wait_thr  = wait_thr

    def reset(self):
        with self._lock:
            self.analytics.reset()

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
            "decode_method":     self.decoder.decode_method,
            "zones_defined":     self.analytics.zones_defined(),
            "queue_zone_ready":  bool(self.analytics.zone_queue),
            "service_zone_ready":bool(self.analytics.zone_service),
            "lines_defined":     self.analytics.lines_defined(),
            "exit_zone_defined": self.analytics.exit_zone_defined(),
            "inference_active":  self.inference_active,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Camera Manager
# ─────────────────────────────────────────────────────────────────────────────
class CameraManager:
    def __init__(self, npu_engine: NPUInferenceEngine):
        self.npu = npu_engine
        self.cameras: Dict[int,CameraStream] = {}
        self._lock = threading.Lock()

    def add(self, name:str, source:str, mode:str="live",
            alert_thr:int=10, wait_thr:float=300.0, inf_fps:float=15.0) -> CameraStream:
        cam = CameraStream(name, source, mode, alert_thr, wait_thr, inf_fps)
        cam.start(self.npu)
        with self._lock: self.cameras[cam.cam_id] = cam
        log.info(f"[Manager] Added cam {cam.cam_id} '{name}'")
        return cam

    def remove(self, cam_id:int):
        with self._lock: cam = self.cameras.pop(cam_id,None)
        if cam: cam.stop()

    def get(self, cam_id:int) -> Optional[CameraStream]:
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
    ph = np.full((480,640,3),30,np.uint8)
    cv2.putText(ph,msg,(20,240),cv2.FONT_HERSHEY_SIMPLEX,0.5,(120,120,120),1)
    _,buf = cv2.imencode(".jpg",ph); return buf.tobytes()


@app.route("/")
def index():
    return send_from_directory(str(_SCRIPT_DIR), "flow-zone-2queus.html")

@app.route("/api/cameras", methods=["GET"])
def list_cameras():
    return jsonify(manager.list())

@app.route("/api/cameras", methods=["POST"])
def add_camera():
    d = request.get_json(force=True)
    name   = d.get("name","Camera").strip() or "Camera"
    source = d.get("source","").strip()
    mode   = d.get("mode","live")
    if not source: return jsonify({"error":"source required"}),400
    at      = int(d.get("alert_threshold",10))
    wt      = float(d.get("wait_time_threshold",300.0))
    inf_fps = float(d.get("inf_fps",15.0))
    cam = manager.add(name,source,mode,at,wt,inf_fps)
    return jsonify(cam.to_dict()),201

@app.route("/api/cameras/<int:cam_id>", methods=["DELETE"])
def remove_camera(cam_id:int):
    manager.remove(cam_id)
    return jsonify({"status":"removed","cam_id":cam_id})

@app.route("/api/cameras/<int:cam_id>/zones", methods=["POST"])
def set_zones(cam_id:int):
    cam=manager.get(cam_id)
    if not cam: abort(404)
    d=request.get_json(force=True)
    cam.set_zones(d.get("zone_queue",[]),d.get("zone_service",[]),d.get("zone_exit",[]))
    if d.get("counter_line") or d.get("exit_line"):
        cam.set_lines(d.get("counter_line",[]), d.get("exit_line",[]))
    at=int(d.get("alert_threshold",cam.analytics.alert_thr))
    wt=float(d.get("wait_time_threshold",cam.analytics.wait_thr))
    cam.set_thresholds(at,wt)
    cam.inference_active=True
    log.info(f"[Cam-{cam_id}] Zones+Lines applied — inference STARTED")
    return jsonify({"status":"ok","cam_id":cam_id,"inference_active":True,"warnings":[]})

@app.route("/api/cameras/<int:cam_id>/lines", methods=["POST"])
def set_lines(cam_id:int):
    cam=manager.get(cam_id)
    if not cam: abort(404)
    d=request.get_json(force=True)
    counter_line = d.get("counter_line",[])
    exit_line    = d.get("exit_line",[])
    cam.set_lines(counter_line, exit_line)
    log.info(f"[Cam-{cam_id}] Lines set — counter={counter_line} exit={exit_line}")
    return jsonify({"status":"ok","cam_id":cam_id,
                    "counter_line":counter_line,"exit_line":exit_line})

@app.route("/api/cameras/<int:cam_id>/lines", methods=["GET"])
def get_lines(cam_id:int):
    cam=manager.get(cam_id)
    if not cam: abort(404)
    a=cam.analytics
    return jsonify({
        "counter_line": a.counter_line,
        "exit_line":    a.exit_line,
        "lines_defined": a.lines_defined(),
    })

@app.route("/api/cameras/<int:cam_id>/zones", methods=["GET"])
def get_zones(cam_id:int):
    cam=manager.get(cam_id)
    if not cam: abort(404)
    a=cam.analytics
    return jsonify({"zone_queue":a.zone_queue,"zone_service":a.zone_service,
                    "zone_exit":a.zone_exit,"alert_threshold":a.alert_thr,
                    "wait_time_threshold":a.wait_thr})

@app.route("/api/cameras/<int:cam_id>/zones", methods=["DELETE"])
def clear_zones(cam_id:int):
    cam=manager.get(cam_id)
    if not cam: abort(404)
    cam.inference_active=False
    cam.set_zones([],[],[])
    cam.analytics.reset()
    log.info(f"[Cam-{cam_id}] Zones cleared — inference STOPPED")
    return jsonify({"status":"zones_cleared","cam_id":cam_id,"inference_active":False})

@app.route("/api/cameras/<int:cam_id>/status")
def cam_status(cam_id:int):
    cam=manager.get(cam_id)
    if not cam: abort(404)
    m=cam.get_metrics(); m.update(cam.to_dict())
    m["npu_device"]=npu_engine.actual_dev
    m["alerts"] = [{k:v for k,v in a.items() if k!="_ts"} for a in m.get("alerts",[])]
    return jsonify(m)

@app.route("/api/cameras/<int:cam_id>/history")
def cam_history(cam_id:int):
    cam=manager.get(cam_id)
    if not cam: abort(404)
    return jsonify(cam.get_history())

@app.route("/api/cameras/<int:cam_id>/history/export")
def cam_history_export(cam_id:int):
    cam=manager.get(cam_id)
    if not cam: abort(404)
    csv_data=cam.get_history_csv()
    fname=f"cam_{cam_id}_{time.strftime('%Y%m%d_%H%M%S')}_history.csv"
    return Response(csv_data,mimetype="text/csv",
                    headers={"Content-Disposition":f"attachment; filename={fname}"})

@app.route("/api/cameras/<int:cam_id>/alerts")
def cam_alerts(cam_id:int):
    cam=manager.get(cam_id)
    if not cam: abort(404)
    return jsonify(cam.analytics.get_alert_history())

@app.route("/api/cameras/<int:cam_id>/alerts/<int:alert_idx>/ack", methods=["POST"])
def ack_alert(cam_id:int, alert_idx:int):
    cam=manager.get(cam_id)
    if not cam: abort(404)
    ok=cam.analytics.acknowledge_alert(alert_idx)
    if not ok: return jsonify({"error":"alert not found"}),404
    return jsonify({"status":"acknowledged","cam_id":cam_id,"alert_idx":alert_idx})

@app.route("/api/cameras/<int:cam_id>/performance")
def cam_performance(cam_id:int):
    cam=manager.get(cam_id)
    if not cam: abort(404)
    perf=cam.analytics.get_counter_performance()
    perf["cam_id"]=cam_id; perf["cam_name"]=cam.name
    perf["exit_zone_defined"]=cam.analytics.exit_zone_defined()
    return jsonify(perf)

@app.route("/api/cameras/<int:cam_id>/reset", methods=["POST"])
def cam_reset(cam_id:int):
    cam=manager.get(cam_id)
    if not cam: abort(404)
    cam.reset(); return jsonify({"status":"reset"})

@app.route("/api/overview")
def overview():
    with manager._lock: cams=list(manager.cameras.values())
    rows=[]; total_queue=0; total_processed=0; total_unacked=0
    for cam in cams:
        m=cam.get_metrics(); perf=cam.analytics.get_counter_performance()
        alerts_all=cam.analytics.get_alert_history()
        unacked=[a for a in alerts_all if not a.get("acked")]
        ql=m.get("queue_length",0)
        total_queue+=ql; total_processed+=m.get("total_processed",0)
        total_unacked+=len(unacked)
        rows.append({
            "cam_id":cam.cam_id,"name":cam.name,"mode":cam.mode,
            "inference_active":cam.inference_active,
            "zones_defined":cam.analytics.zones_defined(),
            "exit_zone_defined":cam.analytics.exit_zone_defined(),
            "decode_method":cam.decoder.decode_method,
            "queue_length":ql,
            "avg_waiting_time":m.get("avg_waiting_time",0),
            "wait_method":m.get("wait_method","none"),
            "open_counter":m.get("open_counter",False),
            "avg_processing_time":m.get("avg_processing_time",0),
            "throughput_per_hour":m.get("throughput_per_hour",0),
            "tput_method":m.get("tput_method","none"),
            "total_processed":m.get("total_processed",0),
            "zone_counts":m.get("zone_counts",{}),
            "queue_entries":m.get("queue_entries",0),
            "queue_exits":m.get("queue_exits",0),
            "unacked_alerts":len(unacked),
            "counter_perf":perf,
            "inf_fps":round(cam._inf_fps,1),
            "inf_ms":round(cam._inf_ms,1),
            "inf_fps_limit":cam.inf_fps_limit,
        })
    return jsonify({
        "cameras":rows,"total_cameras":len(rows),
        "total_queue":total_queue,"total_processed":total_processed,
        "total_unacked_alerts":total_unacked,
        "npu_device":npu_engine.actual_dev,
        "gpu_decode_vaapi":_VAAPI_OK,"gpu_decode_qsv":_QSV_OK,
        "render_device":_RENDER_DEV,
        "timestamp":time.strftime("%H:%M:%S"),
    })

@app.route("/video_feed/<int:cam_id>")
def video_feed(cam_id:int):
    def gen():
        while True:
            cam=manager.get(cam_id)
            jpg=cam.get_jpeg() if cam else None
            if jpg is None: jpg=_placeholder_jpg()
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+jpg+b"\r\n"
            time.sleep(1.0/25)
    return Response(gen(),mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/system")
def sys_stats():
    return jsonify({
        "npu": npu_engine.get_stats(),
        "cameras": len(manager.cameras),
        "gpu_decode": {
            "vaapi":     _VAAPI_OK,
            "qsv":       _QSV_OK,
            "gst_vaapi": _GST_VAAPI_OK,
            "gst_qsv":   _GST_QSV_OK,
            "gst":       _GST_AVAILABLE,
            "device":    _RENDER_DEV,
        },
    })


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Airport Queue Management (Multi-Stream)")
    parser.add_argument("--host",            default="0.0.0.0")
    parser.add_argument("--port",            default=5000, type=int)
    parser.add_argument("--alert-threshold", default=10,   type=int)
    parser.add_argument("--wait-threshold",  default=300.0,type=float)
    args = parser.parse_args()

    npu_engine.start()

    log.info(f"\n🌐  Dashboard → http://localhost:{args.port}")
    log.info(f"    NPU/GPU/CPU Inference via OpenVINO AUTO")
    log.info(f"    GPU Video Decode: GST-VAAPI={_GST_VAAPI_OK}  GST-QSV={_GST_QSV_OK}  "
             f"FFmpeg-VAAPI={_VAAPI_OK}  device={_RENDER_DEV}")
    log.info(f"    Zone-change debounce analytics (no tracking)")
    log.info(f"    Alerts persisted → {_ALERTS_DIR}")
    log.info(f"    Install GPU decode (GStreamer VA-API):")
    log.info(f"      sudo apt install intel-media-va-driver-non-free vainfo")
    log.info(f"      sudo apt install gstreamer1.0-vaapi gstreamer1.0-plugins-bad")
    log.info(f"      sudo apt install gstreamer1.0-plugins-good gstreamer1.0-libav")
    log.info(f"      sudo usermod -aG video,render $USER  # then re-login")
    log.info(f"    APIs:")
    log.info(f"      GET  /api/overview")
    log.info(f"      GET  /api/cameras/<id>/history/export")
    log.info(f"      GET  /api/cameras/<id>/performance")
    log.info(f"      POST /api/cameras/<id>/alerts/<idx>/ack")
    log.info(f"    Add: POST /api/cameras {{name, source, mode, inf_fps}}\n")

    app.run(host=args.host, port=args.port, threaded=True, debug=False)
