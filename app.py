# Import needed libraries
import streamlit as st
import torchvision
from medigan import Generators
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
import time
from datetime import datetime
import io
from PIL import Image
import zipfile

# Page configuration with custom theme
st.set_page_config(
    page_title="MEDIGAN | AI Medical Imaging Studio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for stunning, unconventional design
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Space Grotesk', sans-serif;
    }
    
    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1d3e 50%, #0f1628 100%);
        background-attachment: fixed;
    }
    
    /* Animated gradient header */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-size: 200% 200%;
        animation: gradientShift 8s ease infinite;
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .main-header h1 {
        color: white;
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 4px 20px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.95);
        font-size: 1.3rem;
        margin-top: 0.5rem;
        font-weight: 300;
        position: relative;
        z-index: 1;
    }
    
    /* Glassmorphism sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    section[data-testid="stSidebar"] > div {
        background: transparent;
    }
    
    /* Custom model cards */
    .model-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .model-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
        border-color: rgba(102, 126, 234, 0.6);
    }
    
    /* Stat cards */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    
    .stat-card:hover {
        transform: scale(1.05);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Custom buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6);
    }
    
    /* Image container with glow effect */
    .image-container {
        position: relative;
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1rem;
    }
    
    .image-container:hover {
        transform: scale(1.02);
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.6);
    }
    
    /* Custom selectbox and number input */
    .stSelectbox, .stNumberInput {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
    }
    
    /* Make number input more visible */
    .stNumberInput > div > div > input {
        background: rgba(102, 126, 234, 0.2) !important;
        border: 2px solid rgba(102, 126, 234, 0.5) !important;
        color: white !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        text-align: center !important;
        border-radius: 10px !important;
        padding: 0.8rem !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.5) !important;
    }
    
    /* Number input buttons */
    .stNumberInput button {
        background: rgba(102, 126, 234, 0.3) !important;
        border: 1px solid rgba(102, 126, 234, 0.5) !important;
        color: white !important;
    }
    
    .stNumberInput button:hover {
        background: rgba(102, 126, 234, 0.5) !important;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        border-left: 4px solid #667eea;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
    }
    
    /* Control panel highlight */
    .control-panel {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        border: 2px solid rgba(102, 126, 234, 0.5);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        color: #667eea;
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: white;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom text colors */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
        color: white !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 10px;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    /* Download button special styling */
    .download-btn {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    /* Loading animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading {
        animation: pulse 2s ease-in-out infinite;
    }
    
    /* Highlight box for controls */
    .highlight-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.25) 0%, rgba(118, 75, 162, 0.25) 100%);
        border: 2px solid #667eea;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 0 30px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Define the GAN models with detailed metadata
MODEL_METADATA = {
    "00001_DCGAN_MMG_CALC_ROI": {
        "name": "Mammography Calcification ROI",
        "type": "DCGAN",
        "modality": "Mammography",
        "description": "Generates regions of interest containing calcifications in mammograms",
        "icon": "🔬",
        "color": "#667eea"
    },
    "00002_DCGAN_MMG_MASS_ROI": {
        "name": "Mammography Mass ROI",
        "type": "DCGAN",
        "modality": "Mammography",
        "description": "Generates regions of interest containing masses in mammograms",
        "icon": "🎯",
        "color": "#764ba2"
    },
    "00003_CYCLEGAN_MMG_DENSITY_FULL": {
        "name": "Mammography Density (Full)",
        "type": "CycleGAN",
        "modality": "Mammography",
        "description": "Generates full mammography images with varying breast densities",
        "icon": "🌐",
        "color": "#f093fb"
    },
    "00004_PIX2PIX_MMG_MASSES_W_MASKS": {
        "name": "Mammography Masses with Masks",
        "type": "Pix2Pix",
        "modality": "Mammography",
        "description": "Generates mammography masses with corresponding segmentation masks",
        "icon": "🎭",
        "color": "#f5576c"
    },
    "00019_PGGAN_CHEST_XRAY": {
        "name": "Chest X-Ray",
        "type": "PGGAN",
        "modality": "Chest X-Ray",
        "description": "Generates synthetic chest X-ray images for research and training",
        "icon": "🫁",
        "color": "#4facfe"
    }
}

model_ids = list(MODEL_METADATA.keys())

# Session state initialization
if 'generation_history' not in st.session_state:
    st.session_state.generation_history = []
if 'total_generated' not in st.session_state:
    st.session_state.total_generated = 0
if 'current_images' not in st.session_state:
    st.session_state.current_images = []

def torch_images(num_images, model_id):
    """Generate images using the selected model"""
    generators = Generators()
    dataloader = generators.get_as_torch_dataloader(
        model_id=model_id,
        install_dependencies=True,
        num_samples=num_images,
        prefetch_factor=None,
    )
    images = []
    for batch_idx, data_dict in enumerate(dataloader):
        image_list = []
        for i in data_dict:
            if "sample" in i:
                sample = data_dict.get("sample")
                if sample.dim() == 4:
                    sample = sample.squeeze(0).permute(2, 0, 1)
                sample = to_pil_image(sample).convert("RGB")
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                ])
                sample = transform(sample)
                image_list.append(sample)
            if "mask" in i:
                mask = data_dict.get("mask")
                if mask.dim() == 4:
                    mask = mask.squeeze(0).permute(2, 0, 1)
                mask = to_pil_image(mask).convert("RGB")
                mask = transform(mask)
                image_list.append(mask)
        Grid = make_grid(image_list, nrow=2)
        if Grid.dim() == 4:
            Grid = Grid.squeeze(0)
            if Grid.size(-1) == 1:
                Grid = Grid.squeeze(-1)
            else:
                raise ValueError("Expected a single channel (grayscale) image.")
        img = torchvision.transforms.ToPILImage()(Grid)
        images.append(img)
    return images

def create_zip_download(images, model_id):
    """Create a ZIP file containing all generated images"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for idx, img in enumerate(images):
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            zip_file.writestr(f"medigan_{model_id}_{idx+1}.png", img_buffer.getvalue())
    zip_buffer.seek(0)
    return zip_buffer

def generate_images(num_images, model_id):
    """Enhanced image generation with progress tracking"""
    with st.spinner("🧬 Initializing AI model..."):
        time.sleep(0.5)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(100):
        progress_bar.progress(i + 1)
        if i < 30:
            status_text.text("⚡ Loading neural network...")
        elif i < 60:
            status_text.text("🎨 Generating synthetic medical images...")
        else:
            status_text.text("✨ Finalizing output...")
        time.sleep(0.02)
    
    images = torch_images(num_images, model_id)
    st.session_state.current_images = images
    st.session_state.total_generated += len(images)
    
    # Add to history
    st.session_state.generation_history.append({
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model': MODEL_METADATA[model_id]['name'],
        'count': len(images)
    })
    
    progress_bar.empty()
    status_text.empty()
    
    st.success(f"✅ Successfully generated {len(images)} medical image(s)!")
    
    return images

def main():
    # Header with gradient animation
    st.markdown("""
    <div class="main-header">
        <h1>🧬 MEDIGAN AI Studio</h1>
        <p>Advanced Medical Image Synthesis Platform | Powered by Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration Panel")
        st.markdown("---")
        
        # Statistics at top
        st.markdown("#### 📈 Session Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Generated", st.session_state.total_generated, delta=None)
        with col2:
            st.metric("Models Used", len(set([h['model'] for h in st.session_state.generation_history])))
        
        st.markdown("---")
        
        # Generation history
        if st.session_state.generation_history:
            with st.expander("📜 Generation History", expanded=False):
                for idx, entry in enumerate(reversed(st.session_state.generation_history[-5:])):
                    st.text(f"🕐 {entry['timestamp']}")
                    st.text(f"   {entry['model']} ({entry['count']} imgs)")
                    if idx < len(st.session_state.generation_history) - 1:
                        st.markdown("---")
            st.markdown("---")
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["🎨 Generate", "📚 Model Library", "ℹ️ About"])
    
    with tab1:
        # Generation controls in main area (not sidebar)
        st.markdown("""
        <div class="highlight-box">
            <h2 style="margin-top: 0;">🎯 Generation Controls</h2>
            <p>Configure your medical image generation parameters below</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create two columns for controls
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 🤖 Select AI Model")
            model_id = st.selectbox(
                "Choose the generative model for image synthesis",
                model_ids,
                format_func=lambda x: f"{MODEL_METADATA[x]['icon']} {MODEL_METADATA[x]['name']}",
            )
            
            # Display selected model info
            model_info = MODEL_METADATA[model_id]
            st.markdown(f"""
            <div class="model-card">
                <h4 style="margin-top: 0;">{model_info['icon']} {model_info['name']}</h4>
                <p style="margin: 0.3rem 0;"><strong>Architecture:</strong> {model_info['type']}</p>
                <p style="margin: 0.3rem 0;"><strong>Modality:</strong> {model_info['modality']}</p>
                <p style="margin: 0.3rem 0; font-size: 0.9rem; opacity: 0.9;">{model_info['description']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 📊 Batch Configuration")
            st.markdown("<br>", unsafe_allow_html=True)
            num_images = st.number_input(
                "Number of Images to Generate",
                min_value=1,
                max_value=7,
                value=1,
                step=1,
                help="Generate between 1-7 images in a single batch"
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Generate button with prominent styling
            generate_clicked = st.button("🚀 Generate Images", use_container_width=True, type="primary")
        
        st.markdown("---")
        
        # Check if generation was triggered
        if generate_clicked:
            images = generate_images(num_images, model_id)
            
            # Display images in a grid
            st.markdown("### 🖼️ Generated Medical Images")
            
            # Download all button
            if images:
                zip_buffer = create_zip_download(images, model_id)
                st.download_button(
                    label="📥 Download All Images (ZIP)",
                    data=zip_buffer,
                    file_name=f"medigan_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    use_container_width=True
                )
            
            st.markdown("---")
            
            # Display images in columns
            if len(images) == 1:
                st.image(
                    images[0],
                    caption=f"Generated Image 1 | Model: {MODEL_METADATA[model_id]['name']}",
                    use_container_width=True
                )
            else:
                cols = st.columns(min(3, len(images)))
                for idx, img in enumerate(images):
                    with cols[idx % 3]:
                        st.image(
                            img,
                            caption=f"Image {idx+1} | {MODEL_METADATA[model_id]['name']}",
                            use_container_width=True
                        )
                        
                        # Individual download button
                        img_buffer = io.BytesIO()
                        img.save(img_buffer, format='PNG')
                        img_buffer.seek(0)
                        st.download_button(
                            label=f"💾 Download #{idx+1}",
                            data=img_buffer,
                            file_name=f"medigan_{model_id}_{idx+1}.png",
                            mime="image/png",
                            use_container_width=True,
                            key=f"download_{idx}"
                        )
        else:
            # Welcome screen
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="stat-card">
                    <p class="stat-number">5</p>
                    <p class="stat-label">AI Models</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="stat-card">
                    <p class="stat-number">∞</p>
                    <p class="stat-label">Synthetic Images</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="stat-card">
                    <p class="stat-number">100%</p>
                    <p class="stat-label">Privacy Safe</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-box">
                <h3>🚀 Getting Started</h3>
                <p>Select an AI model above, choose the number of images to generate, and click the <strong>Generate Images</strong> button to create synthetic medical imaging data.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-box">
                <h3>✨ Key Features</h3>
                <ul>
                    <li>🎯 Multiple GAN architectures (DCGAN, CycleGAN, Pix2Pix, PGGAN)</li>
                    <li>🏥 Various medical imaging modalities</li>
                    <li>📦 Batch generation with ZIP download</li>
                    <li>📊 Real-time generation statistics</li>
                    <li>🔒 Privacy-preserving synthetic data</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 🤖 Available AI Models")
        st.markdown("Explore our comprehensive library of medical image generation models.")
        st.markdown("---")
        
        for model_key, model_data in MODEL_METADATA.items():
            with st.expander(f"{model_data['icon']} {model_data['name']}", expanded=False):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**Model ID:** `{model_key}`")
                    st.markdown(f"**Architecture:** {model_data['type']}")
                    st.markdown(f"**Medical Modality:** {model_data['modality']}")
                    st.markdown(f"**Description:** {model_data['description']}")
                with col2:
                    st.markdown(f"<div style='background: {model_data['color']}; width: 100%; height: 100px; border-radius: 10px;'></div>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### 🔬 About MEDIGAN")
        
        st.markdown("""
        <div class="info-box">
            <h4>What is MEDIGAN?</h4>
            <p>MEDIGAN (Medical Image Generation) is a cutting-edge platform that leverages Generative Adversarial Networks (GANs) to create synthetic medical imaging data. This technology enables researchers, clinicians, and AI developers to access high-quality medical images without privacy concerns.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-box">
                <h4>🎯 Use Cases</h4>
                <ul>
                    <li>Training AI diagnostic models</li>
                    <li>Medical education and simulation</li>
                    <li>Algorithm development and testing</li>
                    <li>Data augmentation for rare conditions</li>
                    <li>Privacy-preserving research</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-box">
                <h4>⚡ Technologies</h4>
                <ul>
                    <li>Deep Convolutional GANs</li>
                    <li>CycleGAN architecture</li>
                    <li>Pix2Pix conditional GANs</li>
                    <li>Progressive Growing GANs</li>
                    <li>PyTorch framework</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h4>⚠️ Important Notice</h4>
            <p>These are synthetic images generated by AI models for research and educational purposes. They should not be used for clinical diagnosis or patient care without proper validation and regulatory approval.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()