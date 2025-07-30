// DOM Elements
const frameViewer = document.getElementById("frameViewer");
const currentFrame = document.getElementById("currentFrame");
const playPauseBtn = document.getElementById("playPauseBtn");
const prevFrameBtn = document.getElementById("prevFrameBtn");
const nextFrameBtn = document.getElementById("nextFrameBtn");
const frameSlider = document.getElementById("frameSlider");
const currentFrameNumber = document.getElementById("currentFrameNumber");
const totalFrames = document.getElementById("totalFrames");
const jumpToFrame = document.getElementById("jumpToFrame");
const jumpBtn = document.getElementById("jumpBtn");
const videoUpload = document.getElementById("videoUpload");
const uploadBtn = document.getElementById("uploadBtn");
const downloadBtn = document.getElementById("downloadBtn");
const uploadProgress = document.getElementById("uploadProgress");
const progressBar = uploadProgress.querySelector(".progress-bar");
const progressText = uploadProgress.querySelector(".progress-text");
const framesPreview = document.getElementById("framesPreview");
const captionStartFrame = document.getElementById("captionStartFrame");
const captionEndFrame = document.getElementById("captionEndFrame");
const setStartFrame = document.getElementById("setStartFrame");
const setEndFrame = document.getElementById("setEndFrame");
const captionText = document.getElementById("captionText");
const saveCaption = document.getElementById("saveCaption");
const captionsList = document.getElementById("captionsList");

// State
let currentVideoId = null;
let frames = [];
let captions = [];
let currentFrameIndex = 0;
let isPlaying = false;
let playbackInterval = null;
let fps = 30;

// Utility Functions
function showError(message) {
  alert(message);
  console.error(message);
}

function updateUploadProgress(percent, message) {
  progressBar.style.width = `${percent}%`;
  progressText.textContent = message;
}

// Playback Control
function togglePlayback() {
  if (isPlaying) {
    stopPlayback();
  } else {
    startPlayback();
  }
}

function startPlayback() {
  if (!frames.length) return;

  isPlaying = true;
  playPauseBtn.textContent = "⏸";

  playbackInterval = setInterval(() => {
    if (currentFrameIndex >= frames.length - 1) {
      stopPlayback();
      currentFrameIndex = 0;
    } else {
      currentFrameIndex++;
    }
    updateFrameDisplay();
  }, 1000 / fps);
}

function stopPlayback() {
  isPlaying = false;
  playPauseBtn.textContent = "▶";
  if (playbackInterval) {
    clearInterval(playbackInterval);
    playbackInterval = null;
  }
}

// Frame Navigation
function updateFrameDisplay() {
  if (frames.length === 0) return;

  const frame = frames[currentFrameIndex];
  currentFrame.src = frame.path;
  currentFrameNumber.textContent = currentFrameIndex + 1;
  frameSlider.value = currentFrameIndex;

  // Update frame thumbnails
  const thumbnails = framesPreview.querySelectorAll(".frame-thumbnail");
  thumbnails.forEach((thumb) => thumb.classList.remove("active"));
  thumbnails[currentFrameIndex]?.classList.add("active");

  // Highlight active captions
  highlightActiveCaptions();
}

function highlightActiveCaptions() {
  const captionItems = captionsList.querySelectorAll(".caption-item");
  captionItems.forEach((item) => {
    const startFrame = parseInt(item.dataset.startFrame);
    const endFrame = parseInt(item.dataset.endFrame);
    if (currentFrameIndex >= startFrame && currentFrameIndex <= endFrame) {
      item.classList.add("active");
    } else {
      item.classList.remove("active");
    }
  });
}

// Event Listeners
playPauseBtn.addEventListener("click", togglePlayback);

prevFrameBtn.addEventListener("click", () => {
  stopPlayback();
  if (currentFrameIndex > 0) {
    currentFrameIndex--;
    updateFrameDisplay();
  }
});

nextFrameBtn.addEventListener("click", () => {
  stopPlayback();
  if (currentFrameIndex < frames.length - 1) {
    currentFrameIndex++;
    updateFrameDisplay();
  }
});

frameSlider.addEventListener("input", () => {
  stopPlayback();
  currentFrameIndex = parseInt(frameSlider.value);
  updateFrameDisplay();
});

jumpBtn.addEventListener("click", () => {
  const frameNum = parseInt(jumpToFrame.value) - 1;
  if (frameNum >= 0 && frameNum < frames.length) {
    stopPlayback();
    currentFrameIndex = frameNum;
    updateFrameDisplay();
  }
});

// Frame Preview Grid
function createFramePreview(frame, index) {
  const div = document.createElement("div");
  div.className = "frame-thumbnail";
  if (index === currentFrameIndex) {
    div.classList.add("active");
  }

  div.innerHTML = `
        <img src="${frame.path}" alt="Frame ${index + 1}">
        <div class="frame-number">${index + 1}</div>
    `;

  div.addEventListener("click", () => {
    stopPlayback();
    currentFrameIndex = index;
    updateFrameDisplay();
  });

  return div;
}

function updateFramesPreview() {
  framesPreview.innerHTML = "";
  frames.forEach((frame, index) => {
    const preview = createFramePreview(frame, index);
    framesPreview.appendChild(preview);
  });
}

// Caption Controls
setStartFrame.addEventListener("click", () => {
  captionStartFrame.value = currentFrameIndex;
});

setEndFrame.addEventListener("click", () => {
  captionEndFrame.value = currentFrameIndex;
});

// Video Upload and Download
uploadBtn.addEventListener("click", async () => {
  if (!videoUpload.files.length) {
    showError("Please select a video file first");
    return;
  }

  const file = videoUpload.files[0];
  if (!file.type.startsWith("video/")) {
    showError("Please select a valid video file");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  uploadProgress.style.display = "block";
  downloadBtn.disabled = true;
  updateUploadProgress(0, "Starting upload...");

  try {
    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || "Upload failed");
    }

    updateUploadProgress(50, "Processing video...");

    const data = await response.json();
    currentVideoId = data.filename;
    frames = data.frames;
    fps = data.fps || 30;

    // Setup frame navigation
    frameSlider.max = frames.length - 1;
    totalFrames.textContent = frames.length;
    currentFrameIndex = 0;
    captionStartFrame.max = frames.length - 1;
    captionEndFrame.max = frames.length - 1;

    // Update UI
    updateFrameDisplay();
    updateFramesPreview();
    await loadCaptions();

    downloadBtn.disabled = false;
    updateUploadProgress(100, "Upload complete!");
    setTimeout(() => {
      uploadProgress.style.display = "none";
    }, 2000);
  } catch (error) {
    showError(`Upload failed: ${error.message}`);
    uploadProgress.style.display = "none";
  }
});

downloadBtn.addEventListener("click", async () => {
  if (!currentVideoId) return;

  try {
    const response = await fetch(`/download/${currentVideoId}`);
    if (!response.ok) throw new Error("Download failed");

    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${currentVideoId}_output.mp4`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  } catch (error) {
    showError("Failed to download video: " + error.message);
  }
});

// Captions
async function loadCaptions() {
  if (!currentVideoId) return;

  try {
    const response = await fetch(`/captions/${currentVideoId}`);
    if (!response.ok) throw new Error("Failed to load captions");

    captions = await response.json();
    displayCaptions();
  } catch (error) {
    showError("Error loading captions: " + error.message);
  }
}

function displayCaptions() {
  captionsList.innerHTML = "";

  captions
    .sort((a, b) => a.start_frame - b.start_frame)
    .forEach((caption) => {
      const div = document.createElement("div");
      div.className = "caption-item";
      div.dataset.startFrame = caption.start_frame;
      div.dataset.endFrame = caption.end_frame;
      div.innerHTML = `
                <div class="caption-range">
                    Frames ${caption.start_frame + 1} - ${caption.end_frame + 1}
                </div>
                <div class="caption-text">${caption.text}</div>
            `;

      div.addEventListener("click", () => {
        stopPlayback();
        currentFrameIndex = caption.start_frame;
        updateFrameDisplay();

        // Load caption for editing
        captionStartFrame.value = caption.start_frame;
        captionEndFrame.value = caption.end_frame;
        captionText.value = caption.text;
      });

      captionsList.appendChild(div);
    });
}

saveCaption.addEventListener("click", async () => {
  if (!currentVideoId || !captionText.value.trim()) {
    showError("Please enter caption text");
    return;
  }

  const startFrame = parseInt(captionStartFrame.value);
  const endFrame = parseInt(captionEndFrame.value);

  if (isNaN(startFrame) || isNaN(endFrame) || startFrame > endFrame) {
    showError("Invalid frame range");
    return;
  }

  try {
    const response = await fetch("/save-caption", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        video_id: currentVideoId,
        start_frame: startFrame,
        end_frame: endFrame,
        text: captionText.value.trim(),
      }),
    });

    if (!response.ok) throw new Error("Failed to save caption");

    await loadCaptions();
    captionText.value = "";
  } catch (error) {
    showError("Error saving caption: " + error.message);
  }
});

// Keyboard shortcuts
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "TEXTAREA") return;

  switch (e.code) {
    case "Space":
      e.preventDefault();
      togglePlayback();
      break;
    case "ArrowLeft":
      e.preventDefault();
      prevFrameBtn.click();
      break;
    case "ArrowRight":
      e.preventDefault();
      nextFrameBtn.click();
      break;
    case "KeyS":
      if (e.ctrlKey) {
        e.preventDefault();
        saveCaption.click();
      }
      break;
    case "Enter":
      if (e.ctrlKey) {
        e.preventDefault();
        jumpBtn.click();
      }
      break;
  }
});
