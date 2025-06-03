# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests
import time

# Load the model and processor
processor = AutoImageProcessor.from_pretrained("dima806/medicinal_plants_image_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/medicinal_plants_image_detection")

def classify_medicinal_plant(image_path):
    # Load the local image file
    image = Image.open(image_path)
    
    # Process the image for model input
    inputs = processor(images=image, return_tensors="pt")
    
    # Time the prediction
    time_start = time.time()
    # Get model predictions
    outputs = model(**inputs)
    time_end = time.time()
    
    # Calculate prediction time
    prediction_time = time_end - time_start
    print(f"Time taken for prediction: {prediction_time:.4f} seconds")
    
    # Process predictions
    predictions = outputs.logits.softmax(dim=1)
    
    # Get the predicted class
    predicted_class_idx = predictions.argmax().item()
    predicted_class = model.config.id2label[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx].item()
    
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    
    return {
        "class": predicted_class,
        "confidence": confidence,
        "prediction_time": prediction_time
    }


# import time

# from picamera2 import Picamera2, Preview

# picam2 = Picamera2()
# picam2.start_preview(Preview.QTGL)
# preview_config = picam2.create_preview_configuration()
# capture_config = picam2.create_still_configuration()

# picam2.configure(preview_config)
# picam2.start()
# time.sleep(2)

# image = picam2.switch_mode_and_capture_image(capture_config)
# image.show()


# time.sleep(5)

# picam2.close()
# # download (1).jpeg
# # result = classify_medicinal_plant("download (2).jpeg")

# ...existing code...

def capture_and_classify_plant():
    """
    Captures an image using the camera, classifies it using the model, 
    and returns the predicted class with its confidence.
    """
    from picamera2 import Picamera2, Preview
    import time
    from PIL import Image

    # Initialize the camera
    picam2 = Picamera2()
    # picam2.start_preview(Preview.QTGL)
    preview_config = picam2.create_preview_configuration()
    capture_config = picam2.create_still_configuration()

    # Configure and start the camera
    picam2.configure(preview_config)
    picam2.start()
    time.sleep(2)  # Allow the camera to stabilize

    # Capture the image
    image = picam2.switch_mode_and_capture_image(capture_config)
    picam2.close()

    # Save the image temporarily
    temp_image_path = "captured_image.jpg"
    image.save(temp_image_path)

    # Classify the captured image
    result = classify_medicinal_plant(temp_image_path)

    # Return the result
    return result

# Example usage
if __name__ == "__main__":
    result = capture_and_classify_plant()
    print(f"Class: {result['class']}, Confidence: {result['confidence']:.2%}")