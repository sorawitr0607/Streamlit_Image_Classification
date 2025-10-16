# Import package
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
import pandas as pd
from inference_sdk import InferenceHTTPClient
import numpy as np
#######################################################
# --- Side Bar Config ---
st.set_page_config(layout='wide', page_title="Roboflow Image Classification", page_icon="🤖")
st.sidebar.image('logo.png') 
st.markdown("""
<style>
    div[data-testid="stSidebarUserContent"] img {
        background: #ffffff; /* This is the image's background, not the border's */
        border-radius: 20px; /* Adjust this value for desired roundness */
        padding: 3px; /* Creates space for the border effect */
        background-clip: padding-box; /* Ensures the background gradient only covers the padding area */
        border: 3px solid transparent; /* A transparent border to define the border's space */
        background-image: linear-gradient(to right, darkgrey, white); /* This is the actual gradient for the "border" */
        background-origin: border-box; /* Makes the background start from the border edge */
    }
</style>
""", unsafe_allow_html=True)

#st.sidebar.image(r'C:\Users\Sorawitr\Desktop\streamlit\logo.png',width=180)
st.sidebar.header("About")

st.sidebar.markdown(
    """

App using 🎈[Streamlit](https://streamlit.io/)
""")

st.sidebar.markdown(
    "[Streamlit](https://streamlit.io) is a Python library that allows the creation of interactive, data-driven web applications in Python."
)
st.sidebar.divider()

st.sidebar.header("Resources")
st.sidebar.markdown(
    """
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Cheat sheet](https://docs.streamlit.io/library/cheatsheet)
- [Roboflow Documentation](https://blog.roboflow.com/getting-started-with-roboflow/)
"""
)

st.sidebar.image('roboflow_logo.png') 
#######################################################
# --- Roboflow Configuration ---

ROBOFLOW_API = st.secrets["ROBOFLOW_API"]
ROBOFLOW_MODEL = st.secrets["ROBOFLOW_MODEL"]

#######################################################
# --- Helper Functions ---

def draw_bounding_boxes(image_bytes, detections):
    """
    Draws bounding boxes on an image with thickness and font size
    dynamically scaled to the image dimensions, and attempts to
    prevent text label overlaps.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(image)
    except IOError:
        print("Error: Could not open image from bytes.")
        return None

    img_width, img_height = image.size
    image_diagonal = (img_width**2 + img_height**2) ** 0.5

    BOX_THICKNESS_FACTOR = 0.006
    FONT_SIZE_FACTOR = 0.04

    dynamic_thickness = max(1, int(image_diagonal * BOX_THICKNESS_FACTOR))
    dynamic_font_size = max(12, int(image_diagonal * FONT_SIZE_FACTOR))
    text_padding = max(2, int(dynamic_font_size * 0.15)) # Slightly more padding

    try:
        font = ImageFont.load_default(size=dynamic_font_size)
    except AttributeError:
        font = ImageFont.load_default()
    except IOError:
        print("Warning: Default font not found.")
        font = ImageFont.load_default()

    colors = {"mild": "#FFFF00", "moderate": "#FFA500", "severe": "#FF0000"}

    # --- NEW: Keep track of occupied label regions ---
    occupied_regions = []

    for det in detections:
        center_x = det.get('x')
        center_y = det.get('y')
        width = det.get('width')
        height = det.get('height')
        label = det.get('class')
        
        if not all([center_x, center_y, width, height, label]):
            print(f"Skipping a detection due to missing data: {det}")
            continue

        color = colors.get(label, "#CCCCCC")

        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y + height / 2

        points = ((x1, y1), (x2, y2))
        draw.rectangle(points, outline=color, width=dynamic_thickness)

        confidence = det.get("confidence", 0) * 100
        label_text = f"{label} ({confidence:.1f}%)"

        text_bbox_raw = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox_raw[2] - text_bbox_raw[0]
        text_height = text_bbox_raw[3] - text_bbox_raw[1]

        # Calculate text background dimensions
        bg_width = text_width + (text_padding * 2)
        bg_height = text_height + (text_padding * 2)

        # --- NEW: Smart placement logic ---
        # Function to check for overlap
        def overlaps(rect1, rect2):
            return not (rect1[2] < rect2[0] or rect1[0] > rect2[2] or
                        rect1[3] < rect2[1] or rect1[1] > rect2[3])

        # Attempt to place above, then inside top-left, then inside bottom-left
        potential_positions = [
            # 1. Above the box
            (x1, y1 - bg_height),
            # 2. Inside top-left corner
            (x1, y1),
            # 3. Inside top-right corner (if above is too crowded or goes off-screen)
            (x2 - bg_width, y1),
            # 4. Inside bottom-left corner
            (x1, y2 - bg_height),
            # 5. Inside bottom-right corner
            (x2 - bg_width, y2 - bg_height)
        ]
        
        # Adjust potential positions to be within image bounds if possible
        for i, (px, py) in enumerate(potential_positions):
            px = max(0, min(px, img_width - bg_width))
            py = max(0, min(py, img_height - bg_height))
            potential_positions[i] = (px, py)

        final_pos = None
        for px, py in potential_positions:
            current_label_rect = (px, py, px + bg_width, py + bg_height)
            
            # Check if this position causes overlap with previously placed labels
            is_overlapping = False
            for existing_rect in occupied_regions:
                if overlaps(current_label_rect, existing_rect):
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                final_pos = (px, py)
                break # Found a suitable spot, move on

        # If no non-overlapping spot is found, just place it above the box (or inside if necessary)
        # This is a fallback to ensure the label is always drawn, even if it overlaps.
        if final_pos is None:
            final_pos = potential_positions[0] # Default to first option
            # print(f"Warning: Overlapping label for '{label_text}' at {final_pos}")


        text_bg_x, text_bg_y = final_pos

        # Ensure text is slightly offset within its background for padding
        text_draw_x = text_bg_x + text_padding
        text_draw_y = text_bg_y + text_padding

        text_bg_coords = [
            (text_bg_x, text_bg_y),
            (text_bg_x + bg_width, text_bg_y + bg_height)
        ]
        draw.rectangle(text_bg_coords, fill=color)
        draw.text((text_draw_x, text_draw_y), label_text, fill="black", font=font)
        
        # Add the final placed label's bounding box to occupied regions
        occupied_regions.append(text_bg_coords[0] + text_bg_coords[1]) # Store as (x1, y1, x2, y2)

    return image

def click_button():
    st.session_state.button_analyze = not st.session_state.button_analyze
    st.session_state.button_analyze_disabled = not st.session_state.button_analyze_disabled

def reset_workflow():
    """Resets the app to its initial state"""
    st.session_state.workflow_state_2 = "select"
    st.session_state.analysis_results = None
    st.session_state.annotated_image = None
    st.session_state.button_analyze = False
    st.session_state.button_analyze_disabled = False

#######################################################
# --- Main App Logic ---

def main():

    st.header(":satellite: :violet[Roboflow] Image Analysis with fine-tuned model",divider='violet')

    # Initialize session state to manage workflow
    if 'workflow_state_2' not in st.session_state:
        st.session_state.workflow_state_2 = "select"
        st.session_state.analysis_results = None
        st.session_state.annotated_image = None
        st.session_state.button_analyze = False
        st.session_state.button_analyze_disabled = False
    
    
    
    if st.session_state.workflow_state_2 == "select":
        st.subheader("Select an Image", divider="green")
        uploaded_file = st.file_uploader(
            "Choose an image file", type=["jpg", "jpeg", "png"]
        )
    
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            st.session_state.workflow_state_2 = "preview"
            st.rerun() # Rerun the script to move to the next state
    
    if st.session_state.workflow_state_2 in ["preview", "analysis"]:
        st.subheader("Preview and Analyze", divider="green")
        file = st.session_state.uploaded_file
        file_bytes = file.getvalue()
    
        col1,col2,col3 = st.columns(3)
        with col1:
            st.subheader(":camera_flash: Original Image",divider='blue')
            st.image(file_bytes, caption=file.name, width=400)
        st.button("Analyze Image",disabled= st.session_state.button_analyze,on_click=click_button)
        if st.session_state.button_analyze_disabled:
            with st.spinner("Analyzing..."):
                image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                image_np = np.array(image)
                results = CLIENT.infer(image_np, model_id=ROBOFLOW_MODEL)
                
                if results:
                    st.session_state.analysis_results = results
                    st.session_state.workflow_state_2 = "analysis"
                    st.session_state.button_analyze_disabled = False
                    
                    st.rerun()
                else:
                    st.error("Analysis failed. Check the logs for details.")
    
        if st.session_state.workflow_state_2 == "analysis":
            with col2:
                st.subheader(":mag_right: Analysis Results",divider='blue')
                results = st.session_state.analysis_results
                if results:
                    # Draw boxes and display the new image
                    annotated_image = draw_bounding_boxes(file_bytes, results['predictions'])
                    st.image(annotated_image, caption="Annotated Image", width=400)
                else:
                    st.warning("No analysis results to display.")
            with col3:
                st.subheader(":brain: Label Result",divider='red')
                if results:
                    st.badge("Success", icon=":material/check:", color="green")
                    
                    # st.json(results)
                    df = pd.DataFrame(results['predictions'])
                    if 'class' not in df.columns:
                        st.info("No Accident")
                    else:
                        st.info("Result from ML")
                        st.dataframe(df[['class','confidence']])
                else:
                    st.warning("No analysis results to display.")
    
        if st.button("Start Over"):
            reset_workflow()
            st.rerun()

try:
    
    if not all([ROBOFLOW_MODEL, ROBOFLOW_API]):
         st.error("Credentials or configuration are missing. Please configure them.")
         st.stop()

    # initialize the client
    CLIENT = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=ROBOFLOW_API
    )
    
    
except Exception as e:
    st.error(f"Error initializing clients: {e}")
    st.stop()
    
if __name__ == "__main__":
    main()
