
import streamlit as st
from pathlib import Path
import cv2
import numpy as np
from src.preproc import preprocess_image
from src.models.unet import UNetTiny
import torch

st.set_page_config(page_title='Defect Detector', layout='wide')

@st.cache_resource
def load_model(weights_path='runs/model_epoch_1.pth', device='cpu'):
    model = UNetTiny().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

st.title('Defect Detection — Streamlit Demo')
col1, col2 = st.columns([1,1])

with col1:
    st.header('Input')
    uploaded = st.file_uploader('Upload an image', type=['png','jpg','jpeg'])
    sample_folder = st.checkbox('Use sample folder (data/sample_images)')

with col2:
    st.header('Settings')
    weights = st.text_input('Weights path', 'runs/model_epoch_1.pth')
    device = st.selectbox('Device', ['cpu', 'cuda'])
    run_button = st.button('Run Inference')

model = load_model(weights, 'cpu')

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption='Input', use_column_width=True)
    if run_button:
        proc = preprocess_image(img)
        inp = torch.from_numpy(proc).unsqueeze(0).to('cpu').float()
        with torch.no_grad():
            pred = model(inp)[0,0].cpu().numpy()
        mask = (pred>0.5).astype('uint8')*255
        mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.resize(img, (mask.shape[1], mask.shape[0])), 0.7, mask_color, 0.3, 0)
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption='Overlay', use_column_width=True)

if sample_folder:
    p = Path('data/sample_images')
    imgs = list(p.glob('*.png')) + list(p.glob('*.jpg'))
    if imgs:
        chosen = st.selectbox('Sample images', [str(x) for x in imgs])
        if st.button('Run on sample'):
            img = cv2.imread(chosen)
            proc = preprocess_image(img)
            inp = torch.from_numpy(proc).unsqueeze(0).float()
            with torch.no_grad():
                pred = model(inp)[0,0].numpy()
            mask = (pred>0.5).astype('uint8')*255
            mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(cv2.resize(img, (mask.shape[1], mask.shape[0])), 0.7, mask_color, 0.3, 0)
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption='Overlay', use_column_width=True)
    else:
        st.write('No sample images found in data/sample_images')
