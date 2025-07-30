from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from pathlib import Path
import cv2
import numpy as np
import shutil
import os
import json

# Initialize FastAPI app
app = FastAPI(title="Sign Language Video Caption Tool")

# Create directories if they don't exist
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
FRAMES_DIR = BASE_DIR / "frames"
STATIC_DIR = BASE_DIR / "static"

for dir_path in [UPLOAD_DIR, FRAMES_DIR, STATIC_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Mount static files and frames directory
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/frames", StaticFiles(directory=str(FRAMES_DIR)), name="frames")

# Templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Pydantic models for request validation
class CaptionCreate(BaseModel):
    video_id: str
    start_frame: int
    end_frame: int
    text: str

class VideoManager:
    def __init__(self, video_path, frames_dir):
        self.video_path = Path(video_path)
        self.frames_dir = Path(frames_dir)
        self.video_id = self.video_path.stem
        self.frame_dir = self.frames_dir / self.video_id
        self.frame_dir.mkdir(exist_ok=True, parents=True)
        
    def extract_frames(self):
        print(f"Opening video file: {self.video_path}")
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
            
        frame_count = 0
        frames_info = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_path = self.frame_dir / f"frame_{frame_count:06d}.jpg"
            print(f"Saving frame {frame_count} to {frame_path}")
            success = cv2.imwrite(str(frame_path), frame)
            if not success:
                print(f"Failed to save frame {frame_count}")
                continue
            
            frames_info.append({
                "frame_number": frame_count,
                "path": f"/frames/{self.video_id}/frame_{frame_count:06d}.jpg"
            })
            
            frame_count += 1
            
        cap.release()
        print(f"Extracted {frame_count} frames")
        return frames_info, fps

    def add_caption_to_frame(self, frame, caption_text, frame_height, frame_width):
        """Add caption overlay to a single frame"""
        if not caption_text.strip():
            return frame
            
        # Create a copy of the frame to avoid modifying the original
        frame_with_caption = frame.copy()
        
        # Caption styling parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.8, min(frame_width / 800, frame_height / 600))  # Adaptive font size
        font_thickness = max(1, int(font_scale * 2))
        text_color = (255, 255, 255)  # White text
        bg_color = (0, 0, 0)  # Black background
        padding = 10
        
        # Split text into multiple lines if too long
        max_chars_per_line = max(30, frame_width // 20)
        words = caption_text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) <= max_chars_per_line:
                current_line = current_line + " " + word if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        # Calculate text dimensions
        line_heights = []
        line_widths = []
        
        for line in lines:
            (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, font_thickness)
            line_widths.append(text_width)
            line_heights.append(text_height + baseline)
        
        max_text_width = max(line_widths) if line_widths else 0
        total_text_height = sum(line_heights) + (len(lines) - 1) * 5  # 5px spacing between lines
        
        # Position caption at bottom center
        caption_width = max_text_width + 2 * padding
        caption_height = total_text_height + 2 * padding
        
        # Create semi-transparent background
        x_start = (frame_width - caption_width) // 2
        y_start = frame_height - caption_height - 20  # 20px from bottom
        
        # Ensure caption doesn't go off screen
        x_start = max(0, min(x_start, frame_width - caption_width))
        y_start = max(0, min(y_start, frame_height - caption_height))
        
        # Create overlay for semi-transparent background
        overlay = frame_with_caption.copy()
        cv2.rectangle(overlay, 
                     (x_start, y_start), 
                     (x_start + caption_width, y_start + caption_height),
                     bg_color, -1)
        
        # Blend overlay with original frame
        alpha = 0.7  # Background transparency
        cv2.addWeighted(overlay, alpha, frame_with_caption, 1 - alpha, 0, frame_with_caption)
        
        # Add text lines
        current_y = y_start + padding + line_heights[0]
        
        for i, line in enumerate(lines):
            text_x = x_start + (caption_width - line_widths[i]) // 2
            
            cv2.putText(frame_with_caption, line, 
                       (text_x, current_y), 
                       font, font_scale, text_color, font_thickness, cv2.LINE_AA)
            
            if i < len(lines) - 1:
                current_y += line_heights[i] + 5  # 5px spacing between lines
        
        return frame_with_caption

    def create_video_with_captions(self, output_path, captions_data, fps=30):
        """Create video with captions burned in"""
        print(f"Creating video with captions from frames in {self.frame_dir}")
        frame_files = sorted(self.frame_dir.glob("*.jpg"))
        if not frame_files:
            raise ValueError("No frames found")
            
        first_frame = cv2.imread(str(frame_files[0]))
        if first_frame is None:
            raise ValueError("Could not read first frame")
            
        height, width = first_frame.shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Create a mapping of frame numbers to captions
        frame_captions = {}
        for caption in captions_data:
            caption_text = caption.get("text", "")
            start_frame = caption.get("start_frame", 0)
            end_frame = caption.get("end_frame", 0)
            
            for frame_num in range(start_frame, end_frame + 1):
                if frame_num not in frame_captions:
                    frame_captions[frame_num] = []
                frame_captions[frame_num].append(caption_text)
        
        for i, frame_file in enumerate(frame_files):
            frame = cv2.imread(str(frame_file))
            if frame is not None:
                # Check if this frame has captions
                if i in frame_captions:
                    # Combine multiple captions for the same frame
                    combined_caption = " | ".join(frame_captions[i])
                    frame = self.add_caption_to_frame(frame, combined_caption, height, width)
                
                out.write(frame)
            
        out.release()
        print(f"Video with captions created: {output_path}")
        return output_path

    def create_video(self, output_path, fps=30):
        """Create video without captions (original method kept for backward compatibility)"""
        print(f"Creating video from frames in {self.frame_dir}")
        frame_files = sorted(self.frame_dir.glob("*.jpg"))
        if not frame_files:
            raise ValueError("No frames found")
            
        first_frame = cv2.imread(str(frame_files[0]))
        if first_frame is None:
            raise ValueError("Could not read first frame")
            
        height, width = first_frame.shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            if frame is not None:
                out.write(frame)
            
        out.release()
        return output_path

def load_captions():
    caption_file = UPLOAD_DIR / "captions.json"
    if caption_file.exists():
        with open(caption_file, "r", encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_captions(captions):
    caption_file = UPLOAD_DIR / "captions.json"
    with open(caption_file, "w", encoding='utf-8') as f:
        json.dump(captions, f, indent=4, ensure_ascii=False)

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    try:
        # Create a unique filename
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ['.mp4', '.avi', '.mov', '.mkv']:
            raise ValueError("Unsupported video format. Please use MP4, AVI, MOV, or MKV.")
            
        video_path = UPLOAD_DIR / f"upload_{os.urandom(8).hex()}{file_ext}"
        print(f"Saving uploaded file to: {video_path}")
        
        # Save uploaded video
        with video_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        try:
            # Extract frames
            video_manager = VideoManager(video_path, FRAMES_DIR)
            frames_info, fps = video_manager.extract_frames()
            
            if not frames_info:
                raise ValueError("No frames were extracted from the video")
            
            # Save video info
            video_info = {
                "filename": video_path.stem,
                "frame_count": len(frames_info),
                "frames": frames_info,
                "fps": fps if fps > 0 else 30
            }
            
            return video_info
            
        except Exception as e:
            print(f"Error during frame extraction: {str(e)}")
            if video_path.exists():
                video_path.unlink()
            raise
            
    except Exception as e:
        print(f"Upload error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/save-caption")
async def save_caption(caption: CaptionCreate):
    try:
        captions = load_captions()
        if caption.video_id not in captions:
            captions[caption.video_id] = []
            
        # Check if caption for this frame range exists
        existing_caption = next(
            (c for c in captions[caption.video_id] 
             if c["start_frame"] == caption.start_frame and 
             c["end_frame"] == caption.end_frame),
            None
        )
        
        if existing_caption:
            existing_caption["text"] = caption.text
        else:
            captions[caption.video_id].append({
                "start_frame": caption.start_frame,
                "end_frame": caption.end_frame,
                "text": caption.text
            })
        
        save_captions(captions)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/captions/{video_id}")
async def get_captions(video_id: str):
    captions = load_captions()
    return captions.get(video_id, [])

@app.get("/download/{video_id}")
async def download_video(video_id: str):
    try:
        video_frames_dir = FRAMES_DIR / video_id
        if not video_frames_dir.exists():
            raise HTTPException(status_code=404, detail="Video frames not found")
        
        # Load captions for this video
        captions = load_captions()
        video_captions = captions.get(video_id, [])
        
        output_path = UPLOAD_DIR / f"{video_id}_with_captions.mp4"
        
        # Create VideoManager with the correct video_id
        video_manager = VideoManager(f"{video_id}.mp4", FRAMES_DIR)
        # Manually set the correct paths since we're not using a real video file
        video_manager.video_id = video_id
        video_manager.frame_dir = video_frames_dir
        
        # Use the new method that includes captions
        video_manager.create_video_with_captions(output_path, video_captions)
        
        return FileResponse(
            path=output_path,
            filename=f"{video_id}_with_captions.mp4",
            media_type="video/mp4"
        )
    except Exception as e:
        print(f"Error creating video with captions: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)