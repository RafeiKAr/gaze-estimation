
"""
Personalization Data Recording Script - Corner-Biased Random Startpoints
=======================================================================

Records webcam video + gaze target positions for personalization sessions.

Protocol per 5-second segment:
  - 0.5s WAIT:   Point stationary at (x0, y0) - user finds & fixates
  - 4.5s MOVE:   Point moves at constant velocity, bounces off walls
  - Then jumps to new random startpoint (90% corners, 10% uniform)

Startpoint distribution:
  - 90% in corner regions (10% screen width × 10% screen height each)
  - 10% uniform random across entire screen

User inputs at start:
  - record_number: Session identifier (e.g., "19")
  - duration_sec:  Total recording duration in seconds (e.g., 120)

Outputs (saved in ./personalization/<record_number>/):
  - <record_number>_video.mp4        : Webcam recording (30 FPS)
  - <record_number>_position.txt     : Frame-level positions (segment_id, frame, x, y, phase)
  - <record_number>_subject.txt      : Session metadata & protocol description
  - <record_number>_trajectory.png   : Trajectory plot (1 point/sec)
  - norm_labels.csv                  : Normalized labels for training (generated separately)

Author: Master Thesis - Data-Efficient Personalization for Webcam-Based Eye Tracking
"""

import pygame
import random
import math
import cv2
import matplotlib.pyplot as plt
import warnings
import os
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# Startpoint Generation: 90% corners (10% W × 10% H each), 10% uniform
# ============================================================
def generate_start_position(width, height, radius, corner_probability=0.90, corner_size=0.10):
    """
    Generate a start position with 90% probability in corner regions.
    
    Each corner region: corner_size * width × corner_size * height
    Default: 10% × 10% = 1% screen area per corner → 4% total corner area
    90% of starts in 4% area = very high corner density
    """
    if random.random() < corner_probability:
        # 90%: Pick one of 4 corners (equal weight)
        corner = random.choice(["top_left", "top_right", "bottom_left", "bottom_right"])
        
        corner_w = width * corner_size
        corner_h = height * corner_size
        
        if corner == "top_left":
            x = random.randint(radius, int(corner_w))
            y = random.randint(radius, int(corner_h))
        elif corner == "top_right":
            x = random.randint(int(width - corner_w), width - radius)
            y = random.randint(radius, int(corner_h))
        elif corner == "bottom_left":
            x = random.randint(radius, int(corner_w))
            y = random.randint(int(height - corner_h), height - radius)
        else:  # bottom_right
            x = random.randint(int(width - corner_w), width - radius)
            y = random.randint(int(height - corner_h), height - radius)
    else:
        # 10%: Uniform random across entire screen
        x = random.randint(radius, width - radius)
        y = random.randint(radius, height - radius)
    
    return x, y


# ============================================================
# Main Recording Function
# ============================================================
def main():
    # ----- User Inputs -----
    record_number = input("Enter session number (e.g., 19): ").strip()
    duration_sec = int(input("Enter recording duration in seconds (e.g., 120): ").strip())
    
    # ----- Constants -----
    FPS = 30
    RADIUS = 8
    SPEED = 10  # pixels per frame
    
    WAIT_DURATION = 0.5   # seconds
    MOVE_DURATION = 4.5   # seconds
    SEGMENT_DURATION = WAIT_DURATION + MOVE_DURATION  # 5.0 seconds
    
    WAIT_FRAMES = int(WAIT_DURATION * FPS)    # 15 frames
    MOVE_FRAMES = int(MOVE_DURATION * FPS)    # 135 frames
    SEGMENT_FRAMES = WAIT_FRAMES + MOVE_FRAMES  # 150 frames
    
    TOTAL_FRAMES = int(duration_sec * FPS)
    TOTAL_SEGMENTS = TOTAL_FRAMES // SEGMENT_FRAMES
    
    # ----- Camera Setup -----
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera could not be opened.")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera: {frame_width}×{frame_height} @ {camera_fps:.1f} FPS")
    
    # ----- Pygame Setup -----
    pygame.init()
    info = pygame.display.Info()
    # Ubuntu:
    WIDTH = info.current_w - 50
    HEIGHT = info.current_h - 80

    # Windows:
    # WIDTH = info.current_w - 10
    # HEIGHT = info.current_h - 80
    print(f"Screen: {WIDTH}×{HEIGHT}")
    
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Personalization Recording: Session {record_number} ({duration_sec}s)")
    clock = pygame.time.Clock()
    
    # ----- Output Directory -----
    out_dir = Path(f"./{record_number}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # ----- Video Writer -----
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = out_dir / f"{record_number}_video.mp4"
    video_writer = cv2.VideoWriter(str(video_path), fourcc, FPS, (frame_width, frame_height))
    
    # ----- Position Log File -----
    pos_path = out_dir / f"{record_number}_position.txt"
    pos_file = open(pos_path, "w", encoding="utf-8")
    pos_file.write("segment_id,frame,x,y,phase\n")
    
    # ----- Subject Metadata File -----
    subject_path = out_dir / f"{record_number}_subject.txt"
    subject_file = open(subject_path, "w", encoding="utf-8")
    subject_file.write(f"""Personalization Session Metadata
=====================================
Session ID:          {record_number}
Date/Time:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Recording Duration:  {duration_sec} seconds ({TOTAL_FRAMES} frames @ {FPS} FPS)
Expected Segments:   {TOTAL_SEGMENTS} (each {SEGMENT_DURATION:.1f}s = {WAIT_DURATION}s wait + {MOVE_DURATION}s move)

Startpoint Distribution:
  - Corner probability: 90%
  - Corner region size: 10% width × 10% height per corner (4 corners = 4% screen area)
  - Uniform random:     10%

Protocol per Segment ({SEGMENT_DURATION}s):
  1. WAIT phase ({WAIT_DURATION}s, {WAIT_FRAMES} frames):
     - Point appears at new random startpoint (x0, y0)
     - Point remains STATIONARY
     - User: Find and fixate the point
     - Phase label in position.txt: "wait"
  
  2. MOVE phase ({MOVE_DURATION}s, {MOVE_FRAMES} frames):
     - Point moves at constant velocity (speed={SPEED} px/frame)
     - Random initial direction
     - Bounces off screen edges (wall collision)
     - User: Follow the point smoothly
     - Phase label in position.txt: "move"

  3. JUMP: Instant teleport to new random startpoint → back to WAIT phase

User Inputs Required at Start:
  1. Session number (record_number) - used for all output filenames
  2. Recording duration in seconds (duration_sec) - total recording time

Controls During Recording:
  - 's' key: START recording (camera + position logging)
  - 'q' key: QUIT (stops recording, saves files)
  - ESC key: QUIT (from camera window)

Output Files (in ./personalization/{record_number}/):
  - {record_number}_video.mp4      : Webcam recording
  - {record_number}_position.txt   : Frame-level positions (segment_id, frame, x, y, phase)
  - {record_number}_subject.txt    : This metadata file
  - {record_number}_trajectory.png : Trajectory plot (1 point per second)

Note: norm_labels.csv is generated separately from position.txt + video frames
      using the normalization pipeline (screen.json with W/H per frame).
""")
    subject_file.close()
    
    # ----- Recording State -----
    recording = False
    moving = False
    segment_id = 0
    frame_count = 0  # global frame counter
    segment_frame = 0  # frame within current segment (0 to SEGMENT_FRAMES-1)
    phase = "wait"
    
    # Initial position (will be set when recording starts)
    x = WIDTH // 2
    y = HEIGHT // 2
    vx = 0
    vy = 0
    
    # Trajectory storage (for plot: 1 point per second)
    traj_frames = []
    traj_x = []
    traj_y = []
    
    print(f"\n{'='*60}")
    print(f"Session {record_number} ready. Duration: {duration_sec}s ({TOTAL_SEGMENTS} segments)")
    print(f"Press 's' to START recording, 'q' or ESC to QUIT")
    print(f"{'='*60}\n")
    
    # ----- Main Loop -----
    running = True
    while running:
        clock.tick(FPS)
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                # Start recording
                if event.key == pygame.K_s and not recording:
                    recording = True
                    moving = True
                    segment_id = 0
                    frame_count = 0
                    segment_frame = 0
                    phase = "wait"
                    
                    # Generate first startpoint
                    x, y = generate_start_position(WIDTH, HEIGHT, RADIUS)
                    angle = random.uniform(0, 2 * math.pi)
                    vx = SPEED * math.cos(angle)
                    vy = SPEED * math.sin(angle)
                    
                    print(f"\n[START] Recording started. Segment 1: WAIT phase at ({x:.1f}, {y:.1f})")
                
                # Quit
                if event.key == pygame.K_q:
                    running = False
        
        # ESC from camera window also quits
        if cv2.waitKey(1) & 0xFF == 27:
            running = False
        
        # ----- Recording Logic -----
        if recording and moving:
            # Phase transitions within segment
            if segment_frame == 0:
                # New segment starts: WAIT phase at new startpoint
                phase = "wait"
                if segment_id > 0:  # Not the very first segment
                    x, y = generate_start_position(WIDTH, HEIGHT, RADIUS)
                    angle = random.uniform(0, 2 * math.pi)
                    vx = SPEED * math.cos(angle)
                    vy = SPEED * math.sin(angle)
                    print(f"  [Segment {segment_id + 1}] JUMP → WAIT at ({x:.1f}, {y:.1f})")
            
            elif segment_frame == WAIT_FRAMES:
                # WAIT → MOVE transition
                phase = "move"
                print(f"  [Segment {segment_id + 1}] WAIT → MOVE")
            
            # Movement only in MOVE phase
            if phase == "move":
                x += vx
                y += vy
                
                # Wall collision X
                if x - RADIUS <= 0:
                    x = RADIUS
                    vx = abs(vx)
                elif x + RADIUS >= WIDTH:
                    x = WIDTH - RADIUS
                    vx = -abs(vx)
                
                # Wall collision Y
                if y - RADIUS <= 0:
                    y = RADIUS
                    vy = abs(vy)
                elif y + RADIUS >= HEIGHT:
                    y = HEIGHT - RADIUS
                    vy = -abs(vy)
            
            # Log position
            pos_file.write(f"{segment_id},{frame_count},{x:.6f},{y:.6f},{phase}\n")
            
            # Trajectory point (1 per second = every FPS frames)
            if frame_count % FPS == 0:
                traj_frames.append(frame_count)
                traj_x.append(x / 10.0)  # Scale down for plot (like original scripts)
                traj_y.append(y / 10.0)
            
            # Advance counters
            frame_count += 1
            segment_frame += 1
            
            # Segment complete?
            if segment_frame >= SEGMENT_FRAMES:
                segment_id += 1
                segment_frame = 0
                
                # Check if total duration reached
                if frame_count >= TOTAL_FRAMES:
                    print(f"\n[DONE] Recording complete: {frame_count} frames, {segment_id} segments")
                    moving = False
                    recording = False
        
        # ----- Render -----
        screen.fill((0, 0, 0))
        if recording and moving:
            pygame.draw.circle(screen, (255, 0, 0), (int(x), int(y)), RADIUS)
            # Visual indicator for phase
            if phase == "wait":
                pygame.draw.circle(screen, (255, 255, 0), (int(x), int(y)), RADIUS + 4, 2)
        pygame.display.flip()
        
        # ----- Camera Frame -----
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Camera - ESC to Stop", frame)
            if recording and video_writer is not None:
                video_writer.write(frame)
        
        # Safety stop if duration exceeded
        if frame_count >= TOTAL_FRAMES and recording:
            recording = False
            moving = False
    
    # ----- Cleanup -----
    pos_file.close()
    if video_writer is not None:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    
    # ----- Generate Trajectory Plot -----
    if traj_frames:
        plt.figure(figsize=(6, 6))
        plt.plot(traj_x, traj_y, marker="o", markersize=2, linewidth=0.5)
        plt.xlabel("X-Position (scaled /10)")
        plt.ylabel("Y-Position (scaled /10)")
        plt.title(f"Session {record_number} Trajectory (1 point/sec, {len(traj_frames)} points)")
        plt.axis("equal")
        plt.grid(True)
        traj_path = out_dir / f"{record_number}_trajectory.png"
        plt.savefig(traj_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Trajectory plot saved: {traj_path}")
    
    print(f"\n{'='*60}")
    print(f"Session {record_number} complete.")
    print(f"Frames recorded: {frame_count}")
    print(f"Segments: {segment_id}")
    print(f"Outputs in: {out_dir}")
    print(f"  - {record_number}_video.mp4")
    print(f"  - {record_number}_position.txt")
    print(f"  - {record_number}_subject.txt")
    print(f"  - {record_number}_trajectory.png")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()