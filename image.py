import streamlit as st
import numpy as np
import cv2
from PIL import Image

def main():
    st.title("Image Processing with OpenCV and Streamlit")
   
    # Upload image
    uploaded_file = st.file_uploader(r'C:\Users\A3MAX SOFTWARE TECH\A VS CODE\11. CAPSTONE PROJECT_DEPLOYMENT\numpy matplotlib\ele1', type=["jpg", "jpeg", "png"])
   
    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Convert to numpy array
        img_array = np.array(image)

        # Convert to grayscale using OpenCV
        #gray_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        
        #red_image = cv2.cvtColor(img_array, cv2.COLOR_BGRA2YUV_YUY2)
        
        
        #in_image = cv2.cvtColor(img_array, cv2.COLOR_BAYER_BG2BGR_VNG)
        
        #blue_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
        
        if st.button("Convert to Grayscale"):
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            st.image(gray, caption="Grayscale Image", width=300)
            
        if st.button("Convert to Red Channel"):
            red = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            st.image(red, caption="Red Channel Image", width=300)
            
        if st.button("convert to blue channel"):
            blue = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            st.image(blue, caption="Blue Channel Image", width=300)

            # Convert blue image to bytes for download
            img_bytes = blue.tobytes()
            st.download_button(
                label="download image",
                data=img_bytes,
                file_name="blue_image.png",
                mime="image/png"
            )

        # Display grayscale image
        #st.image(gray_image, caption='Grayscale Image', use_column_width=True, channels="GRAY")
        #st.image(red_image, caption='Red Channel Image', use_column_width=True, channels="BGR")
        #st.image(in_image, caption='In Image', use_column_width=True, channels="BGR")
        #st.image(blue_image, caption='Blue Channel Image', use_column_width=True, channels="BGR")
        

if __name__ == "__main__":
    main()