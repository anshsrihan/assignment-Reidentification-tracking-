# assignment
#  Football Player Re-Identification System

A real-time computer vision project that tracks and **re-identifies football players** in single-camera videos. Even if a player leaves the frame or gets occluded by others, this system tries to bring them back with the **same ID** once they're visible again.

 Note: This is not 100% accurate — identity switches can happen, especially when players wear similar jerseys, numbers aren't visible, or change positions frequently. It's a solid base system, not a production-ready solution.

---

# What This Does

- Detects and tracks multiple players at once  
- Maintains **unique player IDs** throughout the video  
- Handles **occlusions**, **re-entries**, and **lighting variations**  
- Re-identifies players who briefly disappear  
- Saves an annotated output video with visual trails  
- Logs useful tracking stats like total players and FPS

#Known Issues
	•	ID Switches: Similar-looking players may get mixed up (~5–10% of time)
	•	False Detections: Referees, shadows, or staff sometimes tagged as players
	•	Missed Players: Goalkeepers or slow-moving players might be ignored
	•	Re-ID Failures: If a player leaves for too long, a new ID might be assigned
	•	Performance Drops: High-res videos slow down processing
	•	Parameter Sensitivity: Needs tuning for different video types 
 
## Requirement

###  Python Libraries

Install required dependencies:
```bash
pip install opencv-python==4.8.1.78
pip install numpy==1.24.3
'''


