import supervision as sv
from ultralytics import YOLO
import gradio as gr
from PIL import Image
import numpy as np
import cv2
# Load YOLO model
model_loaded = YOLO('best.pt')

# Function for detection and annotation
def detect_text_and_figures(image):
    """
    Detects text and figures in the uploaded image and returns annotated image and detection details.
    """
    try:
        # Convert PIL Image to numpy array
        image = np.array(image)

        # Run YOLO inference
        results = model_loaded(image, conf=0.50)[0]

        # Extract detections
        detections = sv.Detections(
            xyxy=results.boxes.xyxy.cpu().numpy(),
            confidence=results.boxes.conf.cpu().numpy(),
            class_id=results.boxes.cls.cpu().numpy().astype(int)
        )

        # Class names
        class_names = model_loaded.names

        # Annotators for bounding boxes and labels
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator(
            text_scale=0.5,
            text_thickness=1,
        )

        # Annotate the image
        annotated_image = box_annotator.annotate(scene=image, detections=detections)
        annotated_image = label_annotator.annotate(
            scene=annotated_image,
            detections=detections,
            labels=[f"{class_names[class_id]} {confidence:.2f}" for class_id, confidence in zip(detections.class_id, detections.confidence)]
        )

        # Convert annotated image back to PIL Image
        annotated_image_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

        # Create detection details as text
        detection_details = []
        detection_details.append(f"Number of detections: {len(detections)}")
        for box, conf, cls in zip(detections.xyxy, detections.confidence, detections.class_id):
            detection_details.append(f"Detection: {class_names[cls]}, Confidence: {conf:.2f}, Coordinates: {box.tolist()}")

        # Return the annotated image and detection details
        return annotated_image_pil, "\n".join(detection_details)

    except Exception as e:
        # Return error message in case of failure
        return None, f"Error during detection: {str(e)}"

# Define Gradio interface
interface = gr.Interface(
    fn=detect_text_and_figures,
    inputs=gr.Image(type='pil', label="Upload an Image"),
    outputs=[
        gr.Image(type="pil", label="Annotated Image"),
        gr.Textbox(label="Detection Details")
    ],
    title="YOLO Text and Figure Detection",
    description="Upload an image of a PDF page, and the YOLO model will detect text and figures. The output includes an annotated image and detailed detection information."
)

# Launch the Gradio app
interface.launch()
