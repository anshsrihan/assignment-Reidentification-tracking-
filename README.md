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

## Requirement

###  Python Libraries

Install required dependencies:
```bash
pip install opencv-python==4.8.1.78
pip install numpy==1.24.3
