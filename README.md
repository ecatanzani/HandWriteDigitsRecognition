# HandWriteDigitsRecognition
CNN model to recognize hand-written digits

This is a Convolutional Neural Network model to recognize hand-written single digits.

The model is built using TensorFlow Keras APIs.

# CNN model

This is the design of the model, using visualkeras:

![model_layered_view](https://user-images.githubusercontent.com/18316072/164393405-79befc05-309f-4626-a3dd-ae61c59d8f15.png)

The core of the model are three blocks of convolutional 2D layers with same padding, interspersed by max pooling layers and dropout (at 30%). After the convolutional blocks there is flatten layer which connects a feed-forward neural network with 50% dropout. The final layer is a softmax for classification.

Here the network performances:

![Unknown](https://user-images.githubusercontent.com/18316072/164395125-49b6c47a-bf72-4e2a-936d-7376416e1c5f.png)

![Unknown-2](https://user-images.githubusercontent.com/18316072/164395425-d9716a1c-3c16-46ed-83a1-b970cb558b5c.png)

# Streamlit web-application

The project includes a streamlit we-application to use the CNN model

https://user-images.githubusercontent.com/18316072/163561101-c4fea31c-7004-4509-a726-6cd8e809e596.mov

