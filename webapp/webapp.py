import streamlit as st
import numpy as np
import visualkeras
from PIL import Image, ImageOps

st.write("""
# CNN - Picture recognition
""")
black_background = st.checkbox("Use black background", disabled=False)
file = st.file_uploader("Please, choose a picture")

if file is not None:

    img = Image.open(file)

    st.write("**This is your image**")
    st.image(img)

    if not black_background:
        st.write("**Enhancing image contrast**")
        img = ImageOps.invert(img.convert('L'))
        st.image(img)

    # load the Keras model
    model = load_model('model_cnn.h5')
    st.write("**This is the Keras CNN model**")
    st.image(visualkeras.layered_view(model, to_file='model_layered_view.png', legend=True))

    image_size = 28

    data = np.array(img.resize((image_size, image_size)).convert('L'))
    st.write("**Image after resizing**")
    st.image(Image.fromarray(data))

    data_proc = np.reshape(data,[-1, image_size, image_size, 1])
    data_proc = data_proc.astype('float32') / 255
    res = model.predict(data_proc)

    #Showing the probabilities for each category
    num_labels = 10
    for category in range(num_labels):
        print(f"Category {category} --> Probability: {res[0][category]*100}%") 

    y = np.argmax(res, axis=-1)

    st.write(f"The picture is a {y[0]}")

