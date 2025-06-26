import cv2
import numpy as np
from collections import defaultdict, deque
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import json
import time


@dataclass
class Player:
    """Represents a tracked player with their features and history"""
    id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    centroid: Tuple[int, int]
    color_hist: np.ndarray  # Color histogram for appearance
    area: float
    aspect_ratio: float
    last_seen: int  # Frame number
    track_history: deque  # History of centroids
    confidence: float = 0.0


class PlayerTracker:
    """Main class for tracking and re-identifying players"""

    def __init__(self, max_disappeared=30, max_distance=100, history_size=10):
        self.next_id = 0
        self.players = {}  # Active players
        self.disappeared = {}  # Players that disappeared
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.history_size = history_size

        # Initialize background subtractor for better detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50
        )

        # Colors for visualization
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (0, 128, 255), (128, 255, 0), (255, 20, 147), (0, 255, 127)
        ]

    def extract_features(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict:
        """Extract features from a player's bounding box"""
        x1, y1, x2, y2 = bbox

        # Ensure coordinates are within frame bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return None

        # Calculate basic geometric features
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = width / height if height > 0 else 0
        centroid = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Extract color histogram (main appearance feature)
        try:
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # Focus on hue and saturation for jersey color
            hist = cv2.calcHist([roi_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
        except:
            hist = np.zeros(3000)  # Fallback

        return {
            'bbox': bbox,
            'centroid': centroid,
            'area': area,
            'aspect_ratio': aspect_ratio,
            'color_hist': hist
        }

    def detect_players(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect potential player regions using background subtraction and contours"""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        h, w = frame.shape[:2]

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter based on area (adjust these values based on your video)
            if area < 500 or area > 15000:  # Typical player size range
                continue

            x, y, w_box, h_box = cv2.boundingRect(contour)

            # Filter based on aspect ratio (players are usually taller than wide)
            aspect_ratio = w_box / h_box if h_box > 0 else 0
            if aspect_ratio > 1.5 or aspect_ratio < 0.2:  # Reasonable human proportions
                continue

            # Filter based on minimum size
            if w_box < 20 or h_box < 40:
                continue

            detections.append((x, y, x + w_box, y + h_box))

        return detections

    def calculate_distance(self, player: Player, features: Dict) -> float:
        """Calculate distance between existing player and new detection"""
        if features is None:
            return float('inf')

        # Centroid distance (weighted heavily)
        centroid_dist = np.sqrt(
            (player.centroid[0] - features['centroid'][0]) ** 2 +
            (player.centroid[1] - features['centroid'][1]) ** 2
        )

        # Appearance distance (color histogram)
        try:
            color_dist = cv2.compareHist(
                player.color_hist, features['color_hist'], cv2.HISTCMP_BHATTACHARYYA
            )
        except:
            color_dist = 1.0

        # Size similarity
        area_diff = abs(player.area - features['area']) / max(player.area, features['area'])
        aspect_diff = abs(player.aspect_ratio - features['aspect_ratio'])

        # Combine distances with weights
        total_distance = (
                centroid_dist * 0.5 +  # Position is most important
                color_dist * 100 * 0.3 +  # Appearance
                area_diff * 50 * 0.1 +  # Size
                aspect_diff * 20 * 0.1  # Shape
        )

        return total_distance

    def update(self, frame: np.ndarray, frame_num: int) -> Dict[int, Player]:
        """Update tracker with new frame"""
        # Detect players in current frame
        detections = self.detect_players(frame)

        # Extract features for each detection
        detection_features = []
        for bbox in detections:
            features = self.extract_features(frame, bbox)
            if features is not None:
                detection_features.append(features)

        # Match detections with existing players
        if len(self.players) == 0:
            # Initialize with first detections
            for features in detection_features:
                self.register_player(features, frame_num)
        else:
            self.match_detections(detection_features, frame_num)

        # Remove players that have been gone too long
        self.cleanup_disappeared(frame_num)

        return self.players.copy()

    def register_player(self, features: Dict, frame_num: int):
        """Register a new player"""
        player = Player(
            id=self.next_id,
            bbox=features['bbox'],
            centroid=features['centroid'],
            color_hist=features['color_hist'],
            area=features['area'],
            aspect_ratio=features['aspect_ratio'],
            last_seen=frame_num,
            track_history=deque(maxlen=self.history_size)
        )
        player.track_history.append(features['centroid'])

        self.players[self.next_id] = player
        self.next_id += 1

    def match_detections(self, detection_features: List[Dict], frame_num: int):
        """Match current detections with existing players"""
        if len(detection_features) == 0:
            # No detections, mark all as disappeared
            for player_id in list(self.players.keys()):
                self.disappeared[player_id] = self.players[player_id]
                del self.players[player_id]
            return

        # Calculate distance matrix
        player_ids = list(self.players.keys())
        distance_matrix = np.zeros((len(player_ids), len(detection_features)))

        for i, player_id in enumerate(player_ids):
            for j, features in enumerate(detection_features):
                distance_matrix[i, j] = self.calculate_distance(self.players[player_id], features)

        # Assign detections to players using Hungarian algorithm (simplified greedy approach)
        used_detections = set()
        used_players = set()

        # Sort by distance and assign
        assignments = []
        for i in range(len(player_ids)):
            for j in range(len(detection_features)):
                assignments.append((distance_matrix[i, j], i, j))

        assignments.sort()

        for distance, player_idx, detection_idx in assignments:
            if player_idx in used_players or detection_idx in used_detections:
                continue

            if distance < self.max_distance:
                # Update existing player
                player_id = player_ids[player_idx]
                features = detection_features[detection_idx]

                self.players[player_id].bbox = features['bbox']
                self.players[player_id].centroid = features['centroid']
                self.players[player_id].area = features['area']
                self.players[player_id].aspect_ratio = features['aspect_ratio']
                self.players[player_id].last_seen = frame_num
                self.players[player_id].track_history.append(features['centroid'])

                # Update appearance gradually
                alpha = 0.1  # Learning rate
                self.players[player_id].color_hist = (
                        (1 - alpha) * self.players[player_id].color_hist +
                        alpha * features['color_hist']
                )

                used_players.add(player_idx)
                used_detections.add(detection_idx)

        # Handle unmatched players (disappeared)
        for i, player_id in enumerate(player_ids):
            if i not in used_players:
                self.disappeared[player_id] = self.players[player_id]
                del self.players[player_id]

        # Handle unmatched detections (new players or re-identification)
        for j, features in enumerate(detection_features):
            if j not in used_detections:
                # Check if this could be a re-appearing player
                reidentified = False
                min_distance = float('inf')
                best_disappeared_id = None

                for disappeared_id, disappeared_player in self.disappeared.items():
                    distance = self.calculate_distance(disappeared_player, features)
                    if distance < min_distance and distance < self.max_distance * 1.5:
                        min_distance = distance
                        best_disappeared_id = disappeared_id

                if best_disappeared_id is not None:
                    # Re-identify disappeared player
                    player = self.disappeared[best_disappeared_id]
                    player.bbox = features['bbox']
                    player.centroid = features['centroid']
                    player.last_seen = frame_num
                    player.track_history.append(features['centroid'])

                    self.players[best_disappeared_id] = player
                    del self.disappeared[best_disappeared_id]
                    reidentified = True

                if not reidentified:
                    # Register as new player
                    self.register_player(features, frame_num)

    def cleanup_disappeared(self, frame_num: int):
        """Remove players that have been disappeared for too long"""
        to_remove = []
        for player_id, player in self.disappeared.items():
            if frame_num - player.last_seen > self.max_disappeared:
                to_remove.append(player_id)

        for player_id in to_remove:
            del self.disappeared[player_id]


def draw_tracks(frame: np.ndarray, players: Dict[int, Player], colors: List[Tuple[int, int, int]]) -> np.ndarray:
    """Draw player tracks and IDs on the frame"""
    result = frame.copy()

    for player_id, player in players.items():
        color = colors[player_id % len(colors)]

        # Draw bounding box
        x1, y1, x2, y2 = player.bbox
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        # Draw player ID
        cv2.putText(result, f'Player {player_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw track history
        if len(player.track_history) > 1:
            points = list(player.track_history)
            for i in range(1, len(points)):
                cv2.line(result, points[i - 1], points[i], color, 2)

        # Draw centroid
        cv2.circle(result, player.centroid, 5, color, -1)

    return result


def main():
    """Main function to run the player tracking system"""
    # Video file path - replace with your video file
    video_path = "15sec_input_720p.mp4"  # Change this to your video file path

    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found!")
        print("Please update the video_path variable with the correct path to your football video.")
        return

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames / fps:.2f} seconds")

    # Initialize tracker
    tracker = PlayerTracker(max_disappeared=30, max_distance=150)

    # Output video writer (optional)
    output_path = "tracked_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Tracking statistics
    tracking_stats = {
        'total_players_detected': 0,
        'max_simultaneous_players': 0,
        'frames_processed': 0
    }

    frame_num = 0
    start_time = time.time()

    print("\nProcessing video... Press 'q' to quit early.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update tracker
        players = tracker.update(frame, frame_num)

        # Update statistics
        tracking_stats['frames_processed'] = frame_num + 1
        tracking_stats['max_simultaneous_players'] = max(
            tracking_stats['max_simultaneous_players'], len(players)
        )

        # Draw tracking results
        result_frame = draw_tracks(frame, players, tracker.colors)

        # Add frame information
        cv2.putText(result_frame, f'Frame: {frame_num}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result_frame, f'Players: {len(players)}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Write frame to output video
        out.write(result_frame)

        # Display frame (optional - comment out for faster processing)
        cv2.imshow('Football Player Tracking', result_frame)

        # Check for early exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_num += 1

        # Print progress
        if frame_num % 30 == 0:  # Every 30 frames
            elapsed = time.time() - start_time
            progress = frame_num / total_frames * 100
            print(f"Progress: {progress:.1f}% ({frame_num}/{total_frames}) - "
                  f"Speed: {frame_num / elapsed:.1f} FPS")

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Final statistics
    processing_time = time.time() - start_time
    tracking_stats['total_players_detected'] = tracker.next_id

    print(f"\nProcessing completed!")
    print(f"Output saved to: {output_path}")
    print(f"\nTracking Statistics:")
    print(f"  Total unique players detected: {tracking_stats['total_players_detected']}")
    print(f"  Max simultaneous players: {tracking_stats['max_simultaneous_players']}")
    print(f"  Frames processed: {tracking_stats['frames_processed']}")
    print(f"  Processing time: {processing_time:.2f} seconds")
    print(f"  Average processing speed: {tracking_stats['frames_processed'] / processing_time:.2f} FPS")


if __name__ == "__main__":
    main()