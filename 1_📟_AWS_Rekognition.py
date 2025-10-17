# Import package
import streamlit as st
import boto3
from PIL import Image, ImageDraw, ImageFont
import io
import json
import pandas as pd


#######################################################
# --- Side Bar Config ---
st.set_page_config(layout='wide', page_title="AWS Image Classification", page_icon="💻")
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
- [AWS Custom Label Doc](https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/what-is.html)
- [Boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html)
"""
)
st.sidebar.image('AWS-Logo.png') 
#######################################################
# --- AWS Configuration ---

AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
AWS_REGION = st.secrets["AWS_REGION"]  
S3_BUCKET_NAME = st.secrets["S3_BUCKET_NAME"] 
LAMBDA_FUNCTION_NAME = st.secrets["LAMBDA_FUNCTION_NAME"] 


#######################################################
# --- Helper Functions ---

def upload_to_s3(file_obj, bucket, object_name):
    """Uploads a file to a specific subfolder in an S3 bucket."""
    # 1. Define the full path including the subfolder
    full_s3_path = f"img_input_test/{object_name}"
    
    try:
        # 2. Use the full path for the upload
        s3_client.upload_fileobj(file_obj, bucket, full_s3_path)
        
        # 3. Return the correctly formatted S3 URI using the same full path
        st.success(f"Successfully uploaded to s3://{bucket}/{full_s3_path}")
        return f"s3://{bucket}/{full_s3_path}"
    
    except Exception as e:
        st.error(f"Error uploading to S3: {e}")
        return None

def analyze_image_with_lambda(bucket, key):
    """
    Invokes the Lambda function and correctly handles all valid responses,
    including an empty list of labels.
    """
    try:
        payload = {
            "S3Object": {
                "Bucket": bucket,
                "Name": key
            }
        }
        response = lambda_client.invoke(
            FunctionName=LAMBDA_FUNCTION_NAME,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload),
        )
        # 1. Read the entire response from Lambda
        response_payload = json.loads(response["Payload"].read().decode("utf-8"))

        # 2. Check for a successful status code
        if response_payload.get("statusCode") == 200:
            # 3. Get the body, which is a STRING
            body_content = response_payload.get("body", "[]")
            
            # 4. Try to parse the string body into a Python list
            if isinstance(body_content, str):
                try:
                    # This correctly turns "[]" into []
                    return json.loads(body_content)
                except json.JSONDecodeError:
                    # If parsing fails, it was an unexpected message. Return empty.
                    print(f"Lambda returned a non-JSON body: {body_content}")
                    return [] 
            
            # If body_content is somehow already a list/dict, return it
            return body_content
        else:
            # Handle cases where Lambda itself reports an error
            st.error(f"Lambda function returned an error: {response_payload.get('body')}")
            return None

    except Exception as e:
        st.error(f"Error invoking Lambda function: {e}")
        return None

def draw_bounding_boxes(image_bytes, detections):
    """
    Draws bounding boxes on the image using detection data.
    This is the Python equivalent of `renderAnnotatedImageTransformer.js`.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size
    
    # --- 2. Dynamic Scaling Setup (from Function 1) ---
    # Calculate scaling factors based on image diagonal for a consistent look
    image_diagonal = (img_width**2 + img_height**2) ** 0.5
    BOX_THICKNESS_FACTOR = 0.006
    FONT_SIZE_FACTOR = 0.03
    
    # Calculate the dynamic size for box lines and fonts
    dynamic_thickness = max(1, int(image_diagonal * BOX_THICKNESS_FACTOR))
    dynamic_font_size = max(12, int(image_diagonal * FONT_SIZE_FACTOR))
    text_padding = max(2, int(dynamic_font_size * 0.1))
    
    try:
        font = ImageFont.load_default(size=dynamic_font_size)
    except AttributeError: # Fallback for older Pillow versions
        font = ImageFont.load_default()
    except IOError:
        print("Warning: Default font not found.")
        font = ImageFont.load_default()
        
    occupied_regions = []
    def overlaps(rect1, rect2):
        """Checks if two rectangles (x1, y1, x2, y2) overlap."""
        return not (rect1[2] < rect2[0] or rect1[0] > rect2[2] or
                    rect1[3] < rect2[1] or rect1[1] > rect2[3])

    # --- 4. Main Drawing Loop ---
    # Use the color cycling from your target function
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']

    for i, det in enumerate(detections):
        color = colors[i % len(colors)]
        
        # Parse the bounding box data (from Function 2's format)
        box = det.get("Geometry", {}).get("BoundingBox", {})
        if not box:
            box = det.get("BoundingBox", {}) # Fallback for different structures
        
        # Proceed only if box data is valid
        if box and all(k in box for k in ["Left", "Top", "Width", "Height"]):
            # Convert relative coordinates to absolute pixel values
            left = img_width * box["Left"]
            top = img_height * box["Top"]
            width = img_width * box["Width"]
            height = img_height * box["Height"]
            
            # Define box corners for clarity
            x1, y1 = left, top
            x2, y2 = left + width, top + height
            
            # Draw the bounding box with dynamic thickness
            draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=dynamic_thickness)
            
            # Prepare the label text
            label = det.get("Name", "Unknown")
            confidence = det.get("Confidence", 0)
            label_text = f"{label} ({confidence:.1f}%)"
            
            # --- 5. Smart Label Placement Logic (from Function 1) ---
            # Calculate the size needed for the text and its background
            text_bbox_raw = draw.textbbox((0, 0), label_text, font=font)
            text_width = text_bbox_raw[2] - text_bbox_raw[0]
            text_height = text_bbox_raw[3] - text_bbox_raw[1]
            bg_width = text_width + (text_padding * 2)
            bg_height = text_height + (text_padding * 2)

            # Define potential positions for the label to be placed
            potential_positions = [
                (x1, y1 - bg_height),      # Above the box
                (x1, y1),                  # Inside top-left
                (x2 - bg_width, y1),       # Inside top-right
                (x1, y2 - bg_height),      # Inside bottom-left
                (x2 - bg_width, y2 - bg_height) # Inside bottom-right
            ]
            
            # Clamp positions to stay within image bounds
            for j, (px, py) in enumerate(potential_positions):
                px = max(0, min(px, img_width - bg_width))
                py = max(0, min(py, img_height - bg_height))
                potential_positions[j] = (px, py)
                
            # Find the first position that doesn't overlap with others
            final_pos = None
            for px, py in potential_positions:
                current_rect = (px, py, px + bg_width, py + bg_height)
                if not any(overlaps(current_rect, r) for r in occupied_regions):
                    final_pos = (px, py)
                    break
            
            # Fallback to the first choice if all positions overlap
            if final_pos is None:
                final_pos = potential_positions[0]

            # Draw the text background and the text itself
            bg_x, bg_y = final_pos
            draw.rectangle(
                [(bg_x, bg_y), (bg_x + bg_width, bg_y + bg_height)],
                fill=color
            )
            draw.text(
                (bg_x + text_padding, bg_y + text_padding),
                label_text,
                fill="black", # Black text is more readable on colorful backgrounds
                font=font
            )
            
            # Add the final label's region to our list to check against for the next loop
            occupied_regions.append((bg_x, bg_y, bg_x + bg_width, bg_y + bg_height))
        else:
            print(f"Skipping a detection due to missing BoundingBox data: {det}")

    return image

def click_button():
    st.session_state.button_analyze = not st.session_state.button_analyze
    st.session_state.button_analyze_disabled = not st.session_state.button_analyze_disabled

def reset_workflow():
    """Resets the app to its initial state"""
    st.session_state.workflow_state = "upload"
    st.session_state.uploaded_file = None
    st.session_state.analysis_results = None
    st.session_state.annotated_image = None
    st.session_state.button_analyze = False
    st.session_state.button_analyze_disabled = False

#######################################################
# --- Main App Logic ---

def main():

    st.header(":streamlit: :orange[AWS] Image Analysis with Rekognition",divider='orange')

    # Initialize session state to manage workflow
    if 'workflow_state' not in st.session_state:
        st.session_state.workflow_state = "upload"
        st.session_state.uploaded_file = None
        st.session_state.analysis_results = None
        st.session_state.annotated_image = None
        st.session_state.button_analyze = False
        st.session_state.button_analyze_disabled = False
    
    
    
    if st.session_state.workflow_state == "upload":
        st.subheader("Welcome to :green[Driver Behavior Analysis] App", divider="green")
        st.markdown('This application leverages a powerful AI model to analyze images of drivers and classify their behavior.')
        st.markdown('The goal is to identify distracted driving and promote **road safety**. 🚗')
        
        st.subheader("How to Use", divider="blue")
        st.markdown('1. **Upload Image**: *Begin by selecting and uploading a clear photo of a driver in a vehicle.*')
        st.markdown("2. **AI Analysis**: *The application will process the image, using a trained model to detect the driver's actions.*")
        st.markdown('3. **Get the Result**: *The app will display the detected behavior as a label along with a :green[confidence score].*')
        
        st.markdown("<span style='font-size: 24px;'>**<u>Example Output**<u> </span>", unsafe_allow_html=True)
        # Assuming your image path is correct for your local run
        try:
            st.image('tutorial.png', width=800)
        except FileNotFoundError:
            st.warning("Warning: Tutorial image not found at the specified path.")
        
        
        st.subheader("What the App Can Detect", divider="blue")
        st.markdown('Our model is trained to recognize **five distinct categories** of driving behavior based on a real-world dataset.')
        
        # Using st.markdown with HTML for more style control
        st.markdown("<p style='font-weight: bold; color: green;'>✅ Safe Driving</p>", unsafe_allow_html=True)
        st.markdown("<p style='padding-left: 20px;'>The driver is attentive, with hands on the steering wheel or gear stick, and focused on the road.</p>", unsafe_allow_html=True)
        
        st.markdown("<p style='font-weight: bold; color: orange;'>⚠️ Turning</p>", unsafe_allow_html=True)
        st.markdown("<p style='padding-left: 20px;'>The driver's head or body is positioned in a way that indicates they are actively making a turn.</p>", unsafe_allow_html=True)
        
        st.markdown("<p style='font-weight: bold; color: red;'>🚫 Texting Phone 📱</p>", unsafe_allow_html=True)
        st.markdown("<p style='padding-left: 20px;'>The driver is looking down at their phone, typing, or interacting with the screen.</p>", unsafe_allow_html=True)
        
        st.markdown("<p style='font-weight: bold; color: red;'>🚫 Talking on Phone</p>", unsafe_allow_html=True)
        st.markdown("<p style='padding-left: 20px;'>The driver is holding a phone to their ear to engage in a conversation.</p>", unsafe_allow_html=True)
        
        st.markdown("<p style='font-weight: bold; color: grey;'>❔ Others</p>", unsafe_allow_html=True)
        st.markdown("<p style='padding-left: 20px;'>This category includes any other potentially distracting activities, such as drinking, sleeping, etc.</p>", unsafe_allow_html=True)
        
        
        st.header(":orange[!! Try Me] ⬇️⬇️⬇️")
        st.subheader("Upload an Image", divider="green")
        uploaded_file = st.file_uploader(
            "Choose an image file", type=["jpg", "jpeg", "png"]
        )
    
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            st.session_state.workflow_state = "preview"
            st.rerun() # Rerun the script to move to the next state
    
    if st.session_state.workflow_state in ["preview", "analysis"]:
        st.subheader("Preview and Analyze", divider="green")
        file = st.session_state.uploaded_file
        file_bytes = file.getvalue()
    
        col1,col2,col3 = st.columns(3)
        with col1:
            st.subheader(":camera_flash: Original Image",divider='blue')
            st.image(file_bytes, caption=file.name, width=400)
        st.button("Analyze Image",disabled= st.session_state.button_analyze,on_click=click_button)
        if st.session_state.button_analyze_disabled:
            with st.spinner("Uploading to S3..."):
                # 1. Upload to S3
                s3_uri = upload_to_s3(io.BytesIO(file_bytes), S3_BUCKET_NAME, file.name)
                st.success(f"Successfully uploaded to {s3_uri}")
            if s3_uri:
                
                # 2. Invoke Lambda for analysis
                bucket, key = s3_uri.replace("s3://", "").split("/", 1)
                with st.spinner("Analyzing with AWS Rekognition..."):
                    results = analyze_image_with_lambda(bucket, key)
                    st.session_state.analysis_results = results
                    st.session_state.workflow_state = "analysis"
                    st.session_state.button_analyze_disabled = False
                    
                    st.rerun()
                    
    
        if st.session_state.workflow_state == "analysis":
            with col2:
                st.subheader(":mag_right: Analysis Results",divider='blue')
                results = st.session_state.analysis_results
                if results:
                    # Draw boxes and display the new image
                    annotated_image = draw_bounding_boxes(file_bytes, results)
                    st.image(annotated_image, caption="Annotated Image", width=400)
                else:
                    st.image(file_bytes, caption="No Label", width=400)
                    # st.warning("No analysis results to display.")
            with col3:
                st.subheader(":brain: Label Result",divider='red')
                if results:
                    st.badge("Success", icon=":material/check:", color="green")
                    st.info("Result from Rekognition")
                    # st.json(results)
                    df = pd.DataFrame(results)
                    st.dataframe(df[['Name','Confidence']])
                else:
                    st.warning("No analysis results to display.")
    
        if st.button("Start Over"):
            reset_workflow()
            st.rerun()

try:
    
    if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET_NAME, LAMBDA_FUNCTION_NAME]):
         st.error("AWS credentials or configuration are missing. Please configure them.")
         st.stop()

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )

    lambda_client = boto3.client(
        "lambda",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )
    
    
except Exception as e:
    st.error(f"Error initializing AWS clients: {e}")
    st.stop()
    
if __name__ == "__main__":
    main()
    




