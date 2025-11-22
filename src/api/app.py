
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
import cv2
import tempfile
from src.preproc import preprocess_image
from src.models.unet import UNetTiny
import torch

app = FastAPI()

# Load model globally (update path as needed)
MODEL_PATH = 'runs/model_epoch_1.pth'
DEVICE = 'cpu'
model = None
def load_model():
    global model
    if model is None:
        m = UNetTiny()
        m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        m.eval()
        model = m
    return model

@app.post('/infer')
async def infer_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    model = load_model()
    proc = preprocess_image(img)
    inp = torch.from_numpy(proc).unsqueeze(0).float()
    with torch.no_grad():
        pred = model(inp)[0,0].numpy()
    mask = (pred>0.5).astype('uint8')*255
    mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.resize(img, (mask.shape[1], mask.shape[0])), 0.7, mask_color, 0.3, 0)
    _, buf = cv2.imencode('.png', overlay)
    return {"status":"ok", "image_bytes": buf.tobytes()}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
