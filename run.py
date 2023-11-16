import streamlit as st
from PIL import Image
import numpy as np
from configs import *

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('AI OCR')
st.subheader('Nhận dạng chữ viết tay')

image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg', 'JPG'])

if st.button("Chuyển"):

    if image_file is not None:
        img = Image.open(image_file)
        img = np.array(img)

        st.subheader('Hình ảnh được tải lên...')
        st.image(image_file, width=450)
        # Load last checkpoint
        model = train_model(
            input_dim=(configs.height, configs.width, 1),
            output_dim=len(configs.vocab),
        )
        model.load_weights('./Checkpoints/cp.ckpt')
        print(image_file)
        with st.spinner('Đang trích xuất thông tin từ ảnh'):
            text = predict(model, img)[0]
        st.subheader('Từ đã được trích xuất ...')
        st.write(text)

    else:
        st.subheader('Hình ảnh không được tìm thấy! Vui lòng tải lên lại file ảnh.')