from ultralytics import YOLO
import os
from Globals import YOLO_MODELS

class YOLO_Assignment:
    def __init__(self):
        self.models_dir = YOLO_MODELS
        os.makedirs(self.models_dir, exist_ok=True)

    def run(self):
        model_path = os.path.join(self.models_dir, "yolo11n.pt")
        
        if not os.path.exists(model_path):
            print(f"Downloading YOLO model to {model_path}...")
            model = YOLO('yolo11n.pt')
            import shutil
            # YOLO downloads to CWD first
            cwd_location = 'yolo11n.pt'
            if os.path.exists(cwd_location):
                shutil.move(cwd_location, model_path)
        
        model = YOLO(model_path)
        project_path = os.path.join(self.models_dir, "runs")
        model.predict(source="https://ultralytics.com/images/bus.jpg", conf=0.25, save=True, project=project_path)

    def load_model(self):
        pass

    def predict(self, image):
        pass