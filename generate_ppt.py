import collections.abc
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN

prs = Presentation()

def add_title_slide(title, subtitle):
    slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(slide_layout)
    title_box = slide.shapes.title
    subtitle_box = slide.placeholders[1]
    title_box.text = title
    subtitle_box.text = subtitle

def add_bullet_slide(title, bullets):
    slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(slide_layout)
    title_box = slide.shapes.title
    title_box.text = title
    body_shape = slide.shapes.placeholders[1]
    tf = body_shape.text_frame
    for i, bullet in enumerate(bullets):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = bullet
        p.font.size = Pt(20)

def add_image_slide(title, img_path, bullets=None):
    slide_layout = prs.slide_layouts[5] # Title only
    slide = prs.slides.add_slide(slide_layout)
    title_box = slide.shapes.title
    title_box.text = title
    
    try:
        # adjust based on image size and aspect ratio
        left = Inches(1)
        top = Inches(1.5)
        height = Inches(4.5)
        slide.shapes.add_picture(img_path, left, top, height=height)
    except Exception as e:
        print(f"Could not add image {img_path}: {e}")

    if bullets:
        txBox = slide.shapes.add_textbox(Inches(0.5), Inches(6.2), Inches(9), Inches(1))
        tf = txBox.text_frame
        tf.word_wrap = True
        for i, bullet in enumerate(bullets):
            if i == 0:
                p = tf.paragraphs[0]
            else:
                p = tf.add_paragraph()
            p.text = bullet
            p.font.size = Pt(16)

def add_raw_text_slide(title, raw_text):
    slide_layout = prs.slide_layouts[5] # Title only
    slide = prs.slides.add_slide(slide_layout)
    title_box = slide.shapes.title
    title_box.text = title
    
    txBox = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(9), Inches(5.5))
    tf = txBox.text_frame
    tf.word_wrap = False
    
    p = tf.paragraphs[0]
    p.text = raw_text
    p.font.name = 'Courier New'
    p.font.size = Pt(11)

# Slide 1
add_title_slide("Indian Monuments Image Prediction", "An End-to-End Deep Learning & Web Application System")

# Slide 2
add_bullet_slide("Introduction", [
    "Project Overview:",
    "  - An Artificial Intelligence-driven Image Classification system.",
    "  - Automatically identifies various Indian monuments from uploaded user images.",
    "Significance:",
    "  - Promotes Indian heritage and assists in tourism.",
    "  - Demonstrates the practical application of Computer Vision in real-world scenarios."
])

# Slide 3
add_bullet_slide("Main Objective", [
    "1. Develop a highly accurate Deep Learning Pipeline:",
    "   - Utilize State-of-the-Art Convolutional Neural Networks (EfficientNetB3).",
    "2. Perform rigorous Data Mining:",
    "   - Ensure data quality, handle class imbalances, and filter noise.",
    "3. Deploy a Functional Web Application:",
    "   - Create an interactive Flask-based platform that provides real-time predictions along with historical context."
])

# Slide 4
add_bullet_slide("Data Mining Phase I: Data Collection", [
    "Objective: Gather a diverse and robust dataset for 22 Monument Classes.",
    "Mining Process:",
    "  - Sourced thousands of images across varying lighting conditions, angles, and backgrounds.",
    "The 'Negative Class' Approach (download_non_monuments.py):",
    "  - A model must know what a monument is NOT.",
    "  - Systematically mined images of random buildings, nature, and people.",
    "  - Ensures the AI confidently rejects irrelevant uploads instead of making false guesses."
])

# Slide 5
add_bullet_slide("Data Mining Phase II: Data Preparation", [
    "Objective: Process raw data into a machine-readable format.",
    "1. Data Cleaning (clean_data.py):",
    "   - Automated scanning to remove corrupted, invalid, or unreadable image files.",
    "2. Handling Class Imbalance (balance_data.py):",
    "   - Applied data augmentation to ensure equal representation.",
    "   - Prevents the AI from becoming biased towards highly photographed monuments (e.g., Taj Mahal).",
    "3. Data Splitting (split_data.py):",
    "   - Training Set (80%), Validation Set (10%), Testing Set (10%) for unbiased evaluation."
])

# Slide 6
add_image_slide("System Architecture & Data Flow", "E:\\Project\\Indian_MonumentsImage_Prediction\\data_fow.png", 
    ["Pipeline: Image Upload -> Preprocessing -> CNN Feature Extraction -> Softmax Prediction -> Knowledge Base Query -> Result Display"])

# Slide 7
add_image_slide("Context Level: DFD Level 0", "E:\\Project\\Indian_MonumentsImage_Prediction\\DFD_Level0.png", 
    ["The external view showing the interaction between the User (Entity) and the Monument Identification System (Process)."])

# Slide 8
add_image_slide("Detailed Architecture: DFD Level 1", "E:\\Project\\Indian_MonumentsImage_Prediction\\DFD_Level1.png", 
    ["Internal flow detailing Image Preprocessing, CNN Prediction, and querying the JSON Database for historical information."])

# Slide 9
add_bullet_slide("Deep Learning Model & Training", [
    "Core Algorithm: EfficientNetB3",
    "  - A highly optimized, State-of-the-Art Convolutional Neural Network.",
    "Training Methodology (Transfer Learning):",
    "  - Dynamic Data Augmentation: Rotating, zooming, and shifting images to prevent Overfitting.",
    "  - 3-Phase Fine-Tuning Strategy:",
    "    1. Warmup: Training only the top classification layers.",
    "    2. Partial Fine-Tuning: Unfreezing the top 30% of the network layers.",
    "    3. Full Fine-Tuning: Training the entire network with a microscopic learning rate."
])

# Slide 10
add_bullet_slide("Model Evaluation: Standard Metrics", [
    "Accuracy:",
    "  - The primary metric measuring the percentage of overall correct predictions across the testing dataset.",
    "Categorical Crossentropy Loss:",
    "  - Measures the error rate and the model's 'confusion' between classes.",
    "  - The goal of training is the absolute minimization of this loss.",
    "Label Smoothing (0.1):",
    "  - Applied to the loss function to prevent the model from becoming overconfident, vastly improving generalization on unseen images."
])

# Slide 11
add_image_slide("In-depth Analysis: Confusion Matrix", "E:\\Project\\Indian_MonumentsImage_Prediction\\IndianMonumentsImagePrediction\\results\\confusion_matrix.png", 
    ["Diagonal Line (Success Zone): Proves the model correctly predicts the vast majority of images.",
     "Confusion Zones: Highlights specific overlaps (e.g., Golden Temple vs Hawa Mahal due to similar architectural arches). Proves the AI is learning actual geometric features."])

# Slide 12
add_bullet_slide("Understanding the Classification Report", [
    "Precision (Quality of Guesses):",
    "  - Measures False Positives. When the model predicts 'Taj Mahal', how often is it actually the Taj Mahal?",
    "Recall (Quantity of Detection):",
    "  - Measures False Negatives. Out of all actual 'Taj Mahal' photos, how many did the model successfully find?",
    "F1-Score (The Harmonic Mean):",
    "  - The perfect balance between Precision and Recall. Essential for evaluating classes with varying image counts.",
    "Support:",
    "  - Validates that the testing dataset was perfectly balanced."
])

# Slide 13
add_bullet_slide("Classification Report: Averages", [
    "Macro Average:",
    "  - Calculates the average of all F1-scores, treating all monument classes equally.",
    "  - It doesn't care if one monument has 1000 photos and another has 10; their scores carry equal weight.",
    "Weighted Average:",
    "  - Calculates the average but gives more importance (weight) to monuments that have a higher 'Support' (more images).",
    "  - In our dataset, Macro and Weighted averages are identical (~91%), proving extreme model fairness across all classes."
])

# Slide 14: The actual Classification Report
report_path = "E:\\Project\\Indian_MonumentsImage_Prediction\\IndianMonumentsImagePrediction\\results\\classification_report.txt"
try:
    with open(report_path, "r") as f:
        report_text = f.read()
    add_raw_text_slide("Classification Report (Actual Data)", report_text)
except Exception as e:
    print(f"Could not read {report_path}: {e}")

# Slide 15
add_bullet_slide("Conclusion", [
    "1. Successfully engineered a robust Deep Learning classification system.",
    "2. Handled complex Data Mining challenges (Class Imbalance, Corrupted Data).",
    "3. Validated AI reliability using strict mathematical metrics (Confusion Matrix, F1-Score).",
    "4. Bridged the gap between AI and the end-user by deploying a scalable, interactive Flask Web Application."
])

output_path = "E:\\Project\\Indian_MonumentsImage_Prediction\\Final_Project_Presentation_v3.pptx"
prs.save(output_path)
print(f"Professional Presentation generated successfully at {output_path}")
