import time


class CamHandler:
    def __init__(self, cam, fps=30):
        self.cam = cam
        self.fps = fps
        self.last_time = time.time()

    def get_frame(self):
        current_time = time.time()
        time_diff = current_time - self.last_time
        if time_diff < 1 / self.fps:
            # Wait for the next frame time
            time.sleep(1 / self.fps - time_diff)
        
        self.last_time = time.time()
        ret, frame = self.cam.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
        return frame
    
    def set_fps(self, fps):
        self.fps = fps

    def release(self):
        self.cam.release()
