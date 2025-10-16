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
    try:
        s3_client.upload_fileobj(file_obj, bucket, object_name)
        return f"s3://{bucket}/{object_name}"
    except Exception as e:
        st.error(f"Error uploading to S3: {e}")
        return None

def analyze_image_with_lambda(bucket, key):

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
        response_payload = json.loads(response["Payload"].read().decode("utf-8"))


        if response_payload.get("statusCode") == 200:
             # Check if the body is a string that needs to be parsed
            body_content = response_payload.get("body", "[]")
            if isinstance(body_content, str):
                return json.loads(body_content)
            return body_content # Assume it's already a list/dict
        else:
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
    
    # Define a list of colors to cycle through for the boxes
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']

    for i, det in enumerate(detections):
        color = colors[i % len(colors)]
        box = det.get("Geometry", {}).get("BoundingBox", {})
        if not box: # Fallback for slightly different label structures
             box = det.get("BoundingBox", {})
        
        if box:
            left = img_width * box["Left"]
            top = img_height * box["Top"]
            width = img_width * box["Width"]
            height = img_height * box["Height"]
            
            points = ((left, top), (left + width, top + height))
            draw.rectangle(points, outline=color, width=3)
            
            label = det.get("Name", "Unknown")
            confidence = det.get("Confidence", 0)
            label_text = f"{label} ({confidence:.1f}%)"
            
            try:
                # Use a default font. For custom fonts, you'd need to provide a .ttf file.
                font = ImageFont.load_default()
            except IOError:
                font = ImageFont.load_default()

            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Position text above the bounding box
            text_location = [left, top - text_height - 5]
            if text_location[1] < 0: # If text goes off-screen, move it inside
                text_location[1] = top + 5

            draw.rectangle([tuple(text_location), (text_location[0] + text_width + 4, text_location[1] + text_height + 4)], fill=color)
            draw.text(tuple(text_location), label_text, fill="white", font=font)

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
                bucket = s3_uri.split('/')[2]
                key = '/'.join(s3_uri.split('/')[3:])
                with st.spinner("Analyzing with AWS Rekognition..."):
                    results = analyze_image_with_lambda(bucket, key)
                    
                    if results:
                        st.session_state.analysis_results = results
                        st.session_state.workflow_state = "analysis"
                        st.session_state.button_analyze_disabled = False
                        
                        st.rerun()
                    else:
                        st.error("Analysis failed. Check the logs for details.")
    
        if st.session_state.workflow_state == "analysis":
            with col2:
                st.subheader(":mag_right: Analysis Results",divider='blue')
                results = st.session_state.analysis_results
                if results:
                    # Draw boxes and display the new image
                    annotated_image = draw_bounding_boxes(file_bytes, results)
                    st.image(annotated_image, caption="Annotated Image", width=400)
                else:
                    st.warning("No analysis results to display.")
            with col3:
                st.subheader(":brain: Label Result",divider='red')
                if results:
                    st.badge("Success", icon=":material/check:", color="green")
                    st.info("Result from Rekognition")
                    # st.json(results)
                    df = pd.DataFrame(results)
                    st.dataframe(df)
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
    




