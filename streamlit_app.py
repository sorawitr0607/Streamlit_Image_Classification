import streamlit as st

pages = {
    "Select Classify Infrastructure": [
        st.Page("1_📟_AWS_Rekognition.py"),
        st.Page("2_👾_Roboflow_ML.py"),
    ],
}

pg = st.navigation(pages)
pg.run()