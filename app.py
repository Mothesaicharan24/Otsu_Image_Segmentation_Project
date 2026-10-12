from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import os
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


def otsu_from_scratch(gray):
    """Calculate Otsu threshold using between-class variance."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = gray.size

    # Total weighted sum of all intensity values
    total_sum = np.dot(np.arange(256), hist)

    weight_background = 0.0
    sum_background = 0.0
    max_variance = -1.0
    best_threshold = 0

    for t in range(256):
        weight_background += hist[t]
        sum_background += t * hist[t]

        if weight_background == 0:
            continue

        weight_foreground = total - weight_background

        if weight_foreground == 0:
            break

        mean_background = sum_background / weight_background
        mean_foreground = (total_sum - sum_background) / weight_foreground

        # Between-class variance:
        # sigma_b^2 = w0 * w1 * (mu0 - mu1)^2
        between_variance = (
            weight_background / total
            * weight_foreground / total
            * (mean_background - mean_foreground) ** 2
        )

        if between_variance > max_variance:
            max_variance = between_variance
            best_threshold = t

    return best_threshold, max_variance


def make_histogram_image(gray, threshold, filename):
    """Create a visual histogram with the Otsu threshold marked."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()

    width, height = 900, 450
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    max_h = max(hist) if max(hist) > 0 else 1

    # Plot histogram
    for x in range(256):
        bar_h = int((hist[x] / max_h) * 350)
        x1 = int(x * (width - 80) / 256) + 40
        x2 = int((x + 1) * (width - 80) / 256) + 40
        y1 = height - 55 - bar_h
        y2 = height - 55
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 40, 40), -1)

    # Otsu threshold line
    tx = int(threshold * (width - 80) / 256) + 40
    cv2.line(canvas, (tx, 30), (tx, height - 55), (0, 0, 255), 3)

    cv2.putText(canvas, "Intensity (0-255)", (width // 2 - 80, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, f"Otsu Threshold = {threshold}", (tx + 10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    cv2.imwrite(filename, canvas)


def calculate_metrics(binary):
    """Calculate simple segmentation statistics."""
    white = int(np.sum(binary == 255))
    black = int(np.sum(binary == 0))
    total = binary.size

    return {
        "foreground_pixels": white,
        "background_pixels": black,
        "foreground_percentage": round((white / total) * 100, 2),
        "background_percentage": round((black / total) * 100, 2)
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/segment", methods=["POST"])
def segment():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Please select an image."}), 400

    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed:
        return jsonify({"error": "Unsupported image format."}), 400

    image_id = uuid.uuid4().hex
    input_name = image_id + ext
    gray_name = image_id + "_gray.png"
    segmented_name = image_id + "_segmented.png"
    hist_name = image_id + "_histogram.png"

    input_path = os.path.join(UPLOAD_FOLDER, input_name)
    gray_path = os.path.join(RESULT_FOLDER, gray_name)
    segmented_path = os.path.join(RESULT_FOLDER, segmented_name)
    hist_path = os.path.join(RESULT_FOLDER, hist_name)

    file.save(input_path)

    image = cv2.imread(input_path)

    if image is None:
        return jsonify({"error": "Could not read the image."}), 400

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold, between_variance = otsu_from_scratch(gray)

    # Binary segmentation using the calculated Otsu threshold
    _, binary = cv2.threshold(
        gray, threshold, 255, cv2.THRESH_BINARY
    )

    cv2.imwrite(gray_path, gray)
    cv2.imwrite(segmented_path, binary)
    make_histogram_image(gray, threshold, hist_path)

    metrics = calculate_metrics(binary)

    return jsonify({
        "original": "/" + input_path.replace("\\", "/"),
        "gray": "/" + gray_path.replace("\\", "/"),
        "segmented": "/" + segmented_path.replace("\\", "/"),
        "histogram": "/" + hist_path.replace("\\", "/"),
        "threshold": threshold,
        "between_variance": round(float(between_variance), 4),
        "width": int(gray.shape[1]),
        "height": int(gray.shape[0]),
        "metrics": metrics
    })


@app.route("/download/<filename>")
def download(filename):
    path1 = os.path.join(RESULT_FOLDER, filename)
    path2 = os.path.join(UPLOAD_FOLDER, filename)

    if os.path.exists(path1):
        return send_file(path1, as_attachment=True)
    if os.path.exists(path2):
        return send_file(path2, as_attachment=True)

    return "File not found", 404


if __name__ == "__main__":
    app.run(debug=True)
