import cv2
import numpy as np
import tensorflow as tf
import os

keras = tf.keras

BASE = os.path.dirname(os.path.abspath(__file__))


class HandShapeFeatureExtractor:

    __single = None

    @staticmethod
    def get_instance():
        if HandShapeFeatureExtractor.__single is None:
            HandShapeFeatureExtractor()

        return HandShapeFeatureExtractor.__single


    def __init__(self):

        if HandShapeFeatureExtractor.__single is not None:
            raise Exception(
                "This Class bears the model, so it is made Singleton"
            )

        # --------------------------------------------------
        # Recreate the CNN architecture
        # --------------------------------------------------

        real_model = keras.Sequential([
            keras.Input(shape=(300, 300, 3)),

            keras.layers.Conv2D(
                filters=24,
                kernel_size=(3, 3),
                activation="relu",
                padding="same",
                name="conv2d"
            ),

            keras.layers.Conv2D(
                filters=24,
                kernel_size=(3, 3),
                activation="relu",
                padding="same",
                name="conv2d_1"
            ),

            keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding="valid",
                name="max_pooling2d"
            ),

            keras.layers.Conv2D(
                filters=12,
                kernel_size=(3, 3),
                activation="relu",
                padding="same",
                name="conv2d_2"
            ),

            keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding="valid",
                name="max_pooling2d_1"
            ),

            keras.layers.Conv2D(
                filters=6,
                kernel_size=(3, 3),
                activation="relu",
                padding="same",
                name="conv2d_3"
            ),

            keras.layers.Flatten(
                name="flatten"
            ),

            keras.layers.Dense(
                17,
                activation="softmax",
                name="dense"
            )
        ])


        # --------------------------------------------------
        # Load trained weights
        # --------------------------------------------------

        model_path = os.path.join(
            BASE,
            "gestures_trained_cnn_model.h5"
        )

        real_model.load_weights(
            model_path
        )


        # --------------------------------------------------
        # Explicitly call the model once.
        #
        # This is needed with newer Keras versions so that
        # the Sequential model has a defined computation
        # graph and input tensors.
        # --------------------------------------------------

        dummy_input = tf.zeros(
            (1, 300, 300, 3)
        )

        real_model(
            dummy_input
        )


        # --------------------------------------------------
        # Create feature-extraction model
        #
        # The final Dense(17) layer is the classifier.
        # We want the layer immediately before it:
        # the penultimate Flatten layer.
        # --------------------------------------------------

        self.model = keras.Model(
            inputs=real_model.inputs,
            outputs=real_model.layers[-2].output
        )


        HandShapeFeatureExtractor.__single = self


    # --------------------------------------------------
    # Preprocess input frame
    # --------------------------------------------------

    @staticmethod
    def __pre_process_input_image(crop):

        try:

            # Resize to model input dimensions
            img = cv2.resize(
                crop,
                (300, 300)
            )

            # Convert to numpy and normalize
            img_arr = np.array(
                img,
                dtype=np.float32
            ) / 255.0

            # Ensure shape:
            #
            # batch x height x width x channels
            #
            # (1, 300, 300, 3)
            img_arr = img_arr.reshape(
                1,
                300,
                300,
                3
            )

            return img_arr

        except Exception as e:

            print(
                "Error preprocessing image:",
                str(e)
            )

            raise


    # --------------------------------------------------
    # Extract penultimate-layer feature vector
    # --------------------------------------------------

    def extract_feature(self, image):

        try:

            img_arr = (
                self.__pre_process_input_image(
                    image
                )
            )

            feature_vector = self.model.predict(
                img_arr,
                verbose=0
            )

            return feature_vector

        except Exception as e:

            print(
                "Error extracting feature:",
                str(e)
            )

            raise
