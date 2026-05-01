import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import time
# Kalman Filter for 2D centroid tracking

class KalmanFilter2D:
    def __init__(self):
        dt = 1.0
        self.kf = cv2.KalmanFilter(4, 2)
        # Transition matrix (constant velocity model)
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float32)
        # Measurement matrix
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        # Process noise
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        # Measurement noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        # Error covariance
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

    def predict(self) -> Tuple[int, int]:
        pred = self.kf.predict()
        return int(pred[0]), int(pred[1])

    def correct(self, cx: int, cy: int) -> Tuple[int, int]:
        meas = np.array([[np.float32(cx)], [np.float32(cy)]])
        corrected = self.kf.correct(meas)
        return int(corrected[0]), int(corrected[1])

    def set_initial_state(self, cx: int, cy: int):
        self.kf.statePre = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
        self.kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)

# Player dataclass
@dataclass
class Player:
    id: int
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2
    centroid: Tuple[int, int]
    color_hist: np.ndarray
    area: float
    aspect_ratio: float
    last_seen: int
    track_history: deque
    kalman: KalmanFilter2D
    missed_frames: int = 0            # consecutive frames without a match
    confidence: float = 1.0


class PlayerTracker:
    """
    Stable player tracker that uses:
    - Background subtraction for detection
    - Kalman filter per track for motion prediction
    - Hungarian algorithm (scipy) for globally optimal assignment
    - Appearance (color histogram) + positional cost
    - A grace period before a track is truly retired
    """

    def __init__(
        self,
        max_disappeared: int = 45,   # frames before a track is deleted
        max_distance: float = 120,   # Euclidean pixel threshold for position
        appearance_weight: float = 0.35,
        history_size: int = 30,
    ):
        self.next_id = 0
        self.players: Dict[int, Player] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.appearance_weight = appearance_weight
        self.history_size = history_size

        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=40, detectShadows=True
        )

        # 12 visually distinct colours
        self.colors = [
            (220,  20,  60), (  0, 200, 100), ( 30, 144, 255), (255, 200,   0),
            (255,   0, 200), (  0, 220, 220), (128,   0, 200), (255, 120,   0),
            ( 50, 150, 255), (160, 255,  50), (255,  80, 120), ( 80, 255, 180),
        ]

    # Feature extraction 
    def _extract_features(self, frame: np.ndarray, bbox: Tuple) -> Optional[dict]:
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        width  = x2 - x1
        height = y2 - y1
        area   = float(width * height)
        ar     = width / height if height > 0 else 0.0
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # HSV colour histogram (hue + saturation channels)
        try:
            hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
        except Exception:
            hist = np.zeros(3000, dtype=np.float32)

        return {
            'bbox':         (x1, y1, x2, y2),
            'centroid':     (cx, cy),
            'area':         area,
            'aspect_ratio': ar,
            'color_hist':   hist,
        }

    # Background subtraction detection 
    def _detect(self, frame: np.ndarray) -> List[Tuple]:
        fg = self.bg_subtractor.apply(frame)

        # Remove shadows 
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  kernel, iterations=1)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h_frame, w_frame = frame.shape[:2]

        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 400 or area > 20000:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            ar = bw / bh if bh > 0 else 0
            if ar > 1.8 or ar < 0.15:
                continue
            if bw < 15 or bh < 30:
                continue
            boxes.append((x, y, x + bw, y + bh))
        return boxes

    #Cost matrix 
    def _cost(self, player: Player, feat: dict) -> float:
        # Predicted centroid from Kalman
        px, py = player.kalman.predict()
        # Reset prediction (we call predict again properly in update)
        # Use last known centroid instead when Kalman hasn't been set yet
        fx, fy = feat['centroid']

        pos_dist = np.sqrt((player.centroid[0] - fx) ** 2 +
                           (player.centroid[1] - fy) ** 2)

        try:
            app_dist = cv2.compareHist(
                player.color_hist, feat['color_hist'],
                cv2.HISTCMP_BHATTACHARYYA
            )
        except Exception:
            app_dist = 1.0

        return (1.0 - self.appearance_weight) * pos_dist + \
               self.appearance_weight * app_dist * 150.0

    # Register a brandnew track
    def _register(self, feat: dict, frame_num: int):
        kf = KalmanFilter2D()
        cx, cy = feat['centroid']
        kf.set_initial_state(cx, cy)

        p = Player(
            id=self.next_id,
            bbox=feat['bbox'],
            centroid=(cx, cy),
            color_hist=feat['color_hist'],
            area=feat['area'],
            aspect_ratio=feat['aspect_ratio'],
            last_seen=frame_num,
            track_history=deque(maxlen=self.history_size),
            kalman=kf,
        )
        p.track_history.append((cx, cy))
        self.players[self.next_id] = p
        self.next_id += 1


    def _update_player(self, player: Player, feat: dict, frame_num: int):
        cx, cy = feat['centroid']
        player.kalman.correct(cx, cy)

        player.bbox          = feat['bbox']
        player.centroid      = (cx, cy)
        player.area          = feat['area']
        player.aspect_ratio  = feat['aspect_ratio']
        player.last_seen     = frame_num
        player.missed_frames = 0
        player.track_history.append((cx, cy))

        # Slow appearance update (exponential moving average)
        alpha = 0.08
        player.color_hist = (
            (1 - alpha) * player.color_hist + alpha * feat['color_hist']
        )

    def update(self, frame: np.ndarray, frame_num: int) -> Dict[int, Player]:
        raw_boxes  = self._detect(frame)
        feats: List[dict] = []
        for box in raw_boxes:
            f = self._extract_features(frame, box)
            if f is not None:
                feats.append(f)

        player_ids  = list(self.players.keys())
        n_players   = len(player_ids)
        n_feats     = len(feats)

        if n_players == 0:
            for f in feats:
                self._register(f, frame_num)
            return self.players.copy()

        if n_feats == 0:
            # Nothing detected this frame — increment miss counter
            for pid in player_ids:
                self.players[pid].missed_frames += 1
            self._cleanup(frame_num)
            return self.players.copy()

        #Build cost matrix 
        # Use Kalman-predicted positions for the cost
        predicted = {}
        for pid in player_ids:
            predicted[pid] = self.players[pid].kalman.predict()

        C = np.full((n_players, n_feats), fill_value=1e6, dtype=np.float64)
        for i, pid in enumerate(player_ids):
            px, py = predicted[pid]
            for j, f in enumerate(feats):
                fx, fy = f['centroid']
                pos_dist = np.sqrt((px - fx) ** 2 + (py - fy) ** 2)
                try:
                    app_dist = cv2.compareHist(
                        self.players[pid].color_hist,
                        f['color_hist'],
                        cv2.HISTCMP_BHATTACHARYYA,
                    )
                except Exception:
                    app_dist = 1.0
                C[i, j] = (
                    (1.0 - self.appearance_weight) * pos_dist +
                    self.appearance_weight * app_dist * 150.0
                )

        # ── Hungarian algorithm ────────────────────────────────────────────
        row_ind, col_ind = linear_sum_assignment(C)

        matched_players  = set()
        matched_feats    = set()

        for r, c in zip(row_ind, col_ind):
            pid = player_ids[r]
            px, py = predicted[pid]
            fx, fy = feats[c]['centroid']
            pos_dist = np.sqrt((px - fx) ** 2 + (py - fy) ** 2)

            # Accept match only if positional distance is reasonable
            if pos_dist > self.max_distance:
                continue

            self._update_player(self.players[pid], feats[c], frame_num)
            matched_players.add(r)
            matched_feats.add(c)

        # Unmatched players = increment miss counter 
        for i, pid in enumerate(player_ids):
            if i not in matched_players:
                self.players[pid].missed_frames += 1
                # Advance Kalman without a measurement
                self.players[pid].kalman.predict()

        #Unmatched detections → new tracks 
        for j, f in enumerate(feats):
            if j not in matched_feats:
                self._register(f, frame_num)

        self._cleanup(frame_num)
        return self.players.copy()

    #Remove stale tracks
    def _cleanup(self, frame_num: int):
        to_del = [
            pid for pid, p in self.players.items()
            if p.missed_frames > self.max_disappeared
        ]
        for pid in to_del:
            del self.players[pid]


# Drawing
def draw_tracks(frame: np.ndarray, players: Dict[int, Player],
                colors: List[Tuple[int, int, int]]) -> np.ndarray:
    result = frame.copy()
    for pid, player in players.items():
        # Only draw if seen recently (not just in grace period)
        if player.missed_frames > 5:
            continue

        color = colors[pid % len(colors)]
        x1, y1, x2, y2 = player.bbox

        # Bounding box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        # ID label with a filled background for readability
        label = f'P{pid}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(result, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(result, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Track trail
        pts = list(player.track_history)
        for k in range(1, len(pts)):
            alpha = k / len(pts)
            thickness = max(1, int(alpha * 3))
            cv2.line(result, pts[k - 1], pts[k], color, thickness)

        # Centroid dot
        cv2.circle(result, player.centroid, 4, color, -1)

    return result


# Entry point
def main():
    video_path = "15sec_input_720p.mp4"

    if not os.path.exists(video_path):
        print(f"Error: '{video_path}' not found.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open '{video_path}'")
        return

    fps          = int(cap.get(cv2.CAP_PROP_FPS))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps} FPS  ({total_frames} frames)")

    tracker = PlayerTracker(
        max_disappeared=45,   # ~1.5 s at 30 fps before a track is dropped
        max_distance=120,
        appearance_weight=0.35,
    )

    output_path = "tracked_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_num  = 0
    start_time = time.time()
    print("Processing … press 'q' to quit early.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        players      = tracker.update(frame, frame_num)
        result_frame = draw_tracks(frame, players, tracker.colors)

        # HUD
        cv2.putText(result_frame, f'Frame: {frame_num}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result_frame, f'Active tracks: {len(players)}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(result_frame)
        cv2.imshow('Football Player Tracking', result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_num += 1
        if frame_num % 30 == 0:
            elapsed  = time.time() - start_time
            progress = frame_num / total_frames * 100
            print(f"  {progress:.1f}%  ({frame_num}/{total_frames})  "
                  f"{frame_num / elapsed:.1f} FPS")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print(f"\nDone! → {output_path}")
    print(f"Unique track IDs assigned : {tracker.next_id}")
    print(f"Processing time           : {elapsed:.1f}s")


if __name__ == "__main__":
    main()