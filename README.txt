OTSUs-Based Automatic Image Segmentation
========================================

PROJECT TYPE:
B.Tech CSE-AIML Digital Image Processing Major Project

FEATURES:
1. Upload an image
2. Convert RGB image to grayscale
3. Generate intensity histogram
4. Calculate Otsu threshold from scratch
5. Calculate between-class variance
6. Perform automatic binary segmentation
7. Display original, grayscale and segmented images
8. Show threshold and image statistics
9. Download segmented image
10. Responsive web interface

TECHNOLOGIES:
- Python
- Flask
- OpenCV
- NumPy
- HTML
- CSS
- JavaScript

IMPORTANT:
The project uses JavaScript for the browser frontend. It does not use Java.
If your college specifically requires Java instead of JavaScript, the backend
would need to be changed to a Java framework such as Spring Boot.

HOW TO RUN:
1. Install Python 3.x.
2. Open terminal inside this project folder.
3. Create a virtual environment (optional):
       python -m venv venv

4. Activate it on Windows:
       venv\Scripts\activate

5. Install libraries:
       pip install -r requirements.txt

6. Run:
       python app.py

7. Open in browser:
       http://127.0.0.1:5000

HOW THE ALGORITHM WORKS:
For every possible threshold t from 0 to 255:
- Divide pixels into background and foreground classes.
- Calculate their probabilities.
- Calculate their means.
- Calculate between-class variance.
- Choose the threshold with maximum between-class variance.

FORMULA:
sigma_b^2 = w0 * w1 * (mu0 - mu1)^2

where:
w0 = probability of background class
w1 = probability of foreground class
mu0 = mean intensity of background
mu1 = mean intensity of foreground

The selected threshold is then used to create a binary image.

PROJECT EXTENSIONS FOR A MAJOR PROJECT:
- Manual threshold comparison
- Adaptive thresholding
- K-means segmentation
- Fuzzy C-means segmentation
- Segmentation quality metrics
- Noise removal using morphological operations
- Edge detection comparison
- Batch image processing
- Result history/database
- Login/admin dashboard
