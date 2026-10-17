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
    max_area=5000,
    min_circularity=0.5
):
    pad_candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter == 0:
            continue

        circularity = (
            4 * 3.14159 * area
        ) / (perimeter * perimeter)

        if circularity < min_circularity:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        pad_candidates.append({
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "area": area,
            "circularity": circularity
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


def mark_deviating_points(image, points):
    points = points.reshape(-1, 2)

    for i in range(1, len(points) - 1):
        x1, y1 = map(int, points[i - 1])
        x2, y2 = map(int, points[i])
        x3, y3 = map(int, points[i + 1])

        result = (
            (y2 - y1) * (x3 - x2)
            - (y3 - y2) * (x2 - x1)
        )

        if result != 0:
            cv2.circle(
                image,
                (x2, y2),
                3,
                (0, 0, 255),
                -1
            )

    return image

def create_copper_mask(image, min_component_area=8):
    """
    Create an initial binary mask of possible copper features.

    White (255): candidate copper regions
    Black (0): background and excluded regions
    """
    import cv2
    import numpy as np

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Separate dark features from the lighter background
    _, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Connect small gaps within candidate copper regions
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Remove regions connected to the image border
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    cleaned = np.zeros_like(mask)

    height, width = mask.shape

    border_labels = set(labels[0, :])
    border_labels.update(labels[height - 1, :])
    border_labels.update(labels[:, 0])
    border_labels.update(labels[:, width - 1])
    border_labels.discard(0)

    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]

        if label in border_labels:
            continue

        if area < min_component_area:
            continue

        cleaned[labels == label] = 255

    return cleaned

def detect_copper_regions(copper_mask, min_area=8):
    """
    Detect connected regions in a binary copper mask.

    Returns region metadata and a visualization.
    """
    import cv2

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            copper_mask,
            connectivity=8
        )
    )

    regions = []
    output = cv2.cvtColor(
        copper_mask,
        cv2.COLOR_GRAY2BGR
    )

    for label in range(1, count):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        regions.append({
            "label": label,
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
            "area": int(area),
            "centroid": tuple(
                centroids[label]
            )
        })

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            1
        )

    return regions, output

def detect_circular_external_contours(
    image,
    min_area=80,
    min_circularity=0.8,
    min_fill=0.8
):
    """
    Detect circular contours using geometric checks only.

    Returns a tuple of:
      - annotated image (BGR copy with circles drawn in blue)
      - list of accepted contours
    """
    import numpy as np

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, bw = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    cnts, hier = cv2.findContours(
        bw,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    circular_contours = []
    output = image.copy()

    for i, c in enumerate(cnts):
        area = cv2.contourArea(c)
        per = cv2.arcLength(c, True)

        if area < min_area or per == 0:
            continue

        circularity = 4 * np.pi * area / per ** 2

        (cx, cy), r = cv2.minEnclosingCircle(c)
        fill = area / (np.pi * r ** 2)

        if circularity > min_circularity and fill > min_fill:
            circular_contours.append(c)
            cv2.circle(
                output,
                (int(cx), int(cy)),
                int(r),
                (255, 0, 0),
                2
            )
            cv2.circle(
                output,
                (int(cx), int(cy)),
                2,
                (0, 255, 255),
                -1
            )

    return output, circular_contours