import cv2


def load_image(path):
    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {path}"
        )

    return image


def preprocess_image(image):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    return gray, blurred, edges


def detect_contours(edges):
    contours, hierarchy = cv2.findContours(
        edges.copy(),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE
    )

    return contours, hierarchy


def get_external_contours(
    contours,
    hierarchy,
    min_area=30
):
    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]

    external_contours = []

    for i, contour in enumerate(contours):

        if cv2.contourArea(contour) < min_area:
            continue

        if hierarchy[i][3] == -1:
            external_contours.append(contour)

    return external_contours


def draw_contours(image, contours):
    output = image.copy()

    cv2.drawContours(
        output,
        contours,
        -1,
        (0, 255, 0),
        2
    )

    return output


def detect_pad_candidates(
    contours,
    min_area=50,
    max_area=5000
):
    pad_candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if min_area <= area <= max_area:

            x, y, w, h = cv2.boundingRect(
                contour
            )

            pad_candidates.append({
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "area": area
            })

    return pad_candidates


def draw_pad_candidates(
    image,
    pad_candidates
):
    output = image.copy()

    for pad in pad_candidates:

        x = pad["x"]
        y = pad["y"]
        w = pad["width"]
        h = pad["height"]

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

    return output