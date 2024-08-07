import streamlit as st
from PIL import Image
import numpy as np
from configs import HEIGHT, WIDTH, VOCAB
import model
import predict
import io

# st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('AI OCR')
st.subheader('Nhận dạng chữ viết tay')

image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg', 'JPG'])

if st.button("Chuyển"):

    if image_file is not None:
        img = Image.open(image_file)
        img = np.array(img)

        st.subheader('Hình ảnh được tải lên...')
        res = st.image(image_file, width=450)

        # Load last checkpoint
        model = model.construct_model(
            input_dim=(HEIGHT, WIDTH, 1),
            output_dim=len(VOCAB),
        )
        model.load_weights('./checkpoint/cp.ckpt')
        print(image_file)
        with st.spinner('Đang trích xuất thông tin từ ảnh'):
            text = predict.predict(model, img)[0]
        st.subheader('Từ đã được trích xuất ...')
        st.write(text)

    else:
        st.subheader('Hình ảnh không được tìm thấy! Vui lòng tải lên lại file ảnh.')