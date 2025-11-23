
import streamlit as st
from pathlib import Path
import cv2
import numpy as np
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.preproc import preprocess_image
from src.models.unet import UNetTiny
import torch
import matplotlib.pyplot as plt

st.set_page_config(page_title='Defect Detector', layout='wide')

@st.cache_resource
def load_model(weights_path='runs/model_epoch_1.pth', device='cpu'):
    model = UNetTiny().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
    model.eval()
    return model

def create_heatmap(image):
    fig, ax = plt.subplots()
    im = ax.imshow(image, cmap='viridis')
    ax.axis('off')
    fig.colorbar(im)
    return fig

def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))

def iou(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / union

st.title('Defect Detection Demo')

def run_and_display(image, model, device, threshold, col_prefix="", sample_img_path=None):
    """Helper function to run inference and display results in columns."""
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.header(f'{col_prefix}Input Image')
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
        with st.expander("Explanation"):
            st.write("This is the original input image.")

    proc = preprocess_image(image)
    inp = torch.from_numpy(proc).unsqueeze(0).to(device).float()
    with torch.no_grad():
        pred = model(inp)[0,0].cpu().numpy()

    with col2:
        st.header(f'{col_prefix}Prediction Heatmap')
        heatmap_fig = create_heatmap(pred)
        st.pyplot(heatmap_fig, use_container_width=True)
        with st.expander("Explanation"):
            st.write("This heatmap represents the raw output probabilities from the U-Net model.")

    mask_pred = (pred>threshold).astype('uint8')
    mask_color = cv2.applyColorMap(mask_pred*255, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.resize(image, (mask_pred.shape[1], mask_pred.shape[0])), 0.7, mask_color, 0.3, 0)

    with col3:
        st.header(f'{col_prefix}Overlay (Detected Defect)')
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
        with st.expander("Explanation"):
            st.write(f"This image overlays the detected defect based on a threshold of {threshold}.")

    # For sample images, we can also show the ground truth and metrics
    if sample_img_path:
        mask_gt_path = str(sample_img_path).replace('img_', 'mask_')
        if Path(mask_gt_path).exists():
            mask_gt = cv2.imread(mask_gt_path, cv2.IMREAD_GRAYSCALE)
            if mask_gt is not None:
                with col4:
                    st.header(f'{col_prefix}Ground Truth Mask')
                    st.image(mask_gt, use_container_width=True)
                    with st.expander("Explanation"):
                        st.write("This is the ground truth mask.")

                mask_gt_binary = (mask_gt > 128).astype('uint8')
                mask_pred_resized = cv2.resize(mask_pred, (mask_gt_binary.shape[1], mask_gt_binary.shape[0]))
                
                dice = dice_coefficient(mask_gt_binary, mask_pred_resized)
                iou_score = iou(mask_gt_binary, mask_pred_resized)

                st.sidebar.metric(f'{col_prefix}Dice Coefficient', f'{dice:.3f}')
                st.sidebar.metric(f'{col_prefix}IoU', f'{iou_score:.3f}')
            else:
                with col4:
                    st.header(f'{col_prefix}Ground Truth Mask')
                    st.warning(f"Could not read mask: {mask_gt_path}")
        else:
            with col4:
                st.header(f'{col_prefix}Ground Truth Mask')
                st.warning(f"Mask not found at: {mask_gt_path}")


def display_comparison_results(image, model_start, model_end, device, threshold, start_epoch, end_epoch, sample_img_path=None):
    """Helper function to run inference for two models and display results side-by-side for comparison."""
    
    st.subheader(f'Comparing Epoch {start_epoch} vs Epoch {end_epoch}')

    # Load and preprocess image (only once)
    proc = preprocess_image(image)
    inp = torch.from_numpy(proc).unsqueeze(0).to(device).float()

    # Run inference for start_epoch model
    with torch.no_grad():
        pred_start = model_start(inp)[0,0].cpu().numpy()

    # Run inference for end_epoch model
    with torch.no_grad():
        pred_end = model_end(inp)[0,0].cpu().numpy()

    # --- Display Input Image ---
    st.header('Input Image')
    col_start, col_end = st.columns(2)
    with col_start:
        st.caption(f'Epoch {start_epoch}')
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col_end:
        st.caption(f'Epoch {end_epoch}')
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

    # --- Display Prediction Heatmap ---
    st.header('Prediction Heatmap')
    col_start, col_end = st.columns(2)
    with col_start:
        st.caption(f'Epoch {start_epoch}')
        heatmap_fig_start = create_heatmap(pred_start)
        st.pyplot(heatmap_fig_start, use_container_width=True)
    with col_end:
        st.caption(f'Epoch {end_epoch}')
        heatmap_fig_end = create_heatmap(pred_end)
        st.pyplot(heatmap_fig_end, use_container_width=True)

    # --- Display Overlay (Detected Defect) ---
    st.header('Overlay (Detected Defect)')
    mask_pred_start = (pred_start > threshold).astype('uint8')
    mask_color_start = cv2.applyColorMap(mask_pred_start * 255, cv2.COLORMAP_JET)
    overlay_start = cv2.addWeighted(cv2.resize(image, (mask_pred_start.shape[1], mask_pred_start.shape[0])), 0.7, mask_color_start, 0.3, 0)

    mask_pred_end = (pred_end > threshold).astype('uint8')
    mask_color_end = cv2.applyColorMap(mask_pred_end * 255, cv2.COLORMAP_JET)
    overlay_end = cv2.addWeighted(cv2.resize(image, (mask_pred_end.shape[1], mask_pred_end.shape[0])), 0.7, mask_color_end, 0.3, 0)

    col_start, col_end = st.columns(2)
    with col_start:
        st.caption(f'Epoch {start_epoch}')
        st.image(cv2.cvtColor(overlay_start, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col_end:
        st.caption(f'Epoch {end_epoch}')
        st.image(cv2.cvtColor(overlay_end, cv2.COLOR_BGR2RGB), use_container_width=True)


    # --- Display Ground Truth Mask and Metrics (if available) ---
    if sample_img_path:
        mask_gt_path = str(sample_img_path).replace('img_', 'mask_')
        if Path(mask_gt_path).exists():
            mask_gt = cv2.imread(mask_gt_path, cv2.IMREAD_GRAYSCALE)
            if mask_gt is not None:
                st.header('Ground Truth Mask')
                col_start, col_end = st.columns(2)
                with col_start:
                    st.caption(f'Epoch {start_epoch}')
                    st.image(mask_gt, use_container_width=True)
                with col_end:
                    st.caption(f'Epoch {end_epoch}')
                    st.image(mask_gt, use_container_width=True)

                # Metrics
                mask_gt_binary = (mask_gt > 128).astype('uint8')
                
                mask_pred_resized_start = cv2.resize(mask_pred_start, (mask_gt_binary.shape[1], mask_gt_binary.shape[0]))
                dice_start = dice_coefficient(mask_gt_binary, mask_pred_resized_start)
                iou_score_start = iou(mask_gt_binary, mask_pred_resized_start)

                mask_pred_resized_end = cv2.resize(mask_pred_end, (mask_gt_binary.shape[1], mask_gt_binary.shape[0]))
                dice_end = dice_coefficient(mask_gt_binary, mask_pred_resized_end)
                iou_score_end = iou(mask_gt_binary, mask_pred_resized_end)

                st.sidebar.subheader('Comparison Metrics')
                st.sidebar.metric(f'Epoch {start_epoch} Dice', f'{dice_start:.3f}')
                st.sidebar.metric(f'Epoch {start_epoch} IoU', f'{iou_score_start:.3f}')
                st.sidebar.metric(f'Epoch {end_epoch} Dice', f'{dice_end:.3f}')
                st.sidebar.metric(f'Epoch {end_epoch} IoU', f'{iou_score_end:.3f}')
            else:
                st.warning(f"Could not read mask: {mask_gt_path}")
        else:
            st.warning(f"Mask not found at: {mask_gt_path}")


# --- Main App Logic ---

# Sidebar for settings
with st.sidebar:
    st.header('Settings')

    # Scan for available models
    run_dir = Path('runs')
    available_models = list(run_dir.glob('model_epoch_*.pth'))
    
    if not available_models:
        st.error("No models found in the 'runs' directory. Please train a model first.")
        st.stop()

    epoch_numbers = sorted([int(f.stem.split('_')[-1]) for f in available_models])
    
    if len(epoch_numbers) > 0:
        if len(epoch_numbers) > 1:
            selected_epochs = st.select_slider(
                'Select Epoch Range for Comparison',
                options=epoch_numbers,
                value=(min(epoch_numbers), max(epoch_numbers))
            )
            start_epoch, end_epoch = selected_epochs
            st.info(f"Selected epochs: {start_epoch} to {end_epoch}")
        else:
            start_epoch = epoch_numbers[0]
            end_epoch = epoch_numbers[0]
            st.info(f"Only one model epoch found: {start_epoch}")
    else:
        st.error("No model epochs found. Please ensure .pth files are in the 'runs' directory.")
        st.stop()

    device = st.selectbox('Device', ['cpu', 'cuda'])
    threshold = st.slider('Prediction Threshold', 0.0, 1.0, 0.5)

    st.header('Input')
    uploaded = st.file_uploader('Upload an image', type=['png','jpg','jpeg'])
    sample_folder = st.checkbox('Use sample folder (data/procedural_images)')


# --- Image and Inference Display ---

if uploaded is not None and not sample_folder:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if st.sidebar.button('Run Inference'):
        if start_epoch == end_epoch:
            weights_path = f'runs/model_epoch_{start_epoch}.pth'
            model = load_model(weights_path, device)
            run_and_display(img, model, device, threshold)
        else:
            st.error("Epoch comparison is only available for sample images.")

if sample_folder:
    p = Path('data/procedural_images')
    imgs = list(p.glob('img_*.png')) + list(p.glob('img_*.jpg'))
    if imgs:
        chosen = st.sidebar.selectbox('Sample images', [str(x) for x in imgs])
        
        if st.sidebar.button('Run on sample'):
            img = cv2.imread(chosen)
            
            if start_epoch == end_epoch:
                weights_path = f'runs/model_epoch_{start_epoch}.pth'
                model = load_model(weights_path, device)
                run_and_display(img, model, device, threshold, sample_img_path=chosen)
            else:
                weights_path_start = f'runs/model_epoch_{start_epoch}.pth'
                model_start = load_model(weights_path_start, device)
                weights_path_end = f'runs/model_epoch_{end_epoch}.pth'
                model_end = load_model(weights_path_end, device)
                display_comparison_results(img, model_start, model_end, device, threshold, start_epoch, end_epoch, sample_img_path=chosen)

    else:
        st.write('No sample images found in data/procedural_images')

