import cv2

from src.image_processor import (
    load_image,
    preprocess_image,
    detect_contours,
    get_external_contours,
    draw_contours,
    detect_pad_candidates,
    draw_pad_candidates
)


# -----------------------------
# 1. Load PCB image
# -----------------------------

image = load_image(
    "input/pcb_sample.jpeg"
)


# -----------------------------
# 2. Preprocess image
# -----------------------------

gray, blurred, edges = preprocess_image(
    image
)


# -----------------------------
# 3. Detect contours
# -----------------------------

contours, hierarchy = detect_contours(
    edges
)


# -----------------------------
# 4. Get external contours
# -----------------------------

external_contours = get_external_contours(
    contours,
    hierarchy
)


# -----------------------------
# 5. Detect pad candidates
# -----------------------------

pad_candidates = detect_pad_candidates(
    contours
)


# -----------------------------
# 6. Print information
# -----------------------------

print("Image loaded successfully!")

print("Original:", image.shape)

print("Grayscale:", gray.shape)

print("Blurred:", blurred.shape)

print("Edges:", edges.shape)

print("Contours detected:", len(contours))

print(
    "External contours:",
    len(external_contours)
)

print(
    "Pad candidates:",
    len(pad_candidates)
)


for i, pad in enumerate(pad_candidates):

    print(
        f"Pad {i + 1}: {pad}"
    )


# -----------------------------
# 7. Draw external contours
# -----------------------------

output = draw_contours(
    image,
    external_contours
)


cv2.imwrite(
    "output/external_contours.jpg",
    output
)


print(
    "Saved: output/external_contours.jpg"
)


# -----------------------------
# 8. Draw pad candidates
# -----------------------------

pad_output = draw_pad_candidates(
    image,
    pad_candidates
)


cv2.imwrite(
    "output/pad_candidates.jpg",
    pad_output
)


print(
    "Saved: output/pad_candidates.jpg"
)