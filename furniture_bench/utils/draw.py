import cv2
import numpy as np
import numpy.typing as npt

from furniture_bench.perception.realsense import RealsenseCam


def draw_tags(
    color_image: npt.NDArray[np.uint8], cam: RealsenseCam, tags
) -> npt.NDArray[np.uint8]:
    draw_img = color_image.copy()
    for tag in tags:
        if tag is None:
            continue
        # Draw boarder of the tag.
        draw_img = draw_bbox(draw_img, tag)
        # Draw x, y, z axis on the image.
        draw_img = draw_axis(draw_img, tag.pose_R, tag.pose_t, cam.intr_mat).copy()

        # Draw id next to the tag.
        draw_img = cv2.putText(
            draw_img,
            str(tag.tag_id),
            org=(
                tag.corners[0, 0].astype(int) + 10,
                tag.corners[0, 1].astype(int) + 10,
            ),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=(0, 0, 255),
        )
    return draw_img


def draw_bbox(color_image: npt.NDArray[np.uint8], tag):
    if tag is None:
        return color_image
    draw_img = color_image.copy()
    for idx in range(len(tag.corners)):
        draw_img = cv2.line(
            draw_img,
            tuple(tag.corners[idx - 1, :].astype(int)),
            tuple(tag.corners[idx, :].astype(int)),
            (0, 255, 0),
        )
    return draw_img


def draw_axis(
    img: npt.NDArray[np.uint8],
    R: npt.NDArray[np.float32],
    t: npt.NDArray[np.float32],
    K: npt.NDArray[np.float32],
    s: float = 0.015,
    d: int = 3,
    rgb=True,
    axis="xyz",
    colors=None,
    trans=False,
    text_label: bool = False,
    draw_arrow: bool = False,
) -> npt.NDArray[np.uint8]:
    """Draw x, y, z axis on the image.

    Args:
        img: Image to draw on.
        R: Rotation matrix.
        t: Translation vector.
        K: Intrinsic matrix.
        s: Length of the axis.
        d: Thickness of the axis.


    Returns:
        Image with the axis drawn.
    """
    draw_img = img.copy()
    # Unit is m
    rotV, _ = cv2.Rodrigues(R)
    # The tag's coordinate frame is centered at the center of the tag,
    # with x-axis to the right, y-axis down, and z-axis into the tag.
    if isinstance(s, float):
        points = np.float32([[s, 0, 0], [0, s, 0], [0, 0, s], [0, 0, 0]]).reshape(-1, 3)
    else:
        # list
        points = np.float32(
            [[s[0], 0, 0], [0, s[1], 0], [0, 0, s[2]], [0, 0, 0]]
        ).reshape(-1, 3)

    axis_points, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
    a0 = np.array((int(axis_points[0][0][0]), int(axis_points[0][0][1])))
    a1 = np.array((int(axis_points[1][0][0]), int(axis_points[1][0][1])))
    a2 = np.array((int(axis_points[2][0][0]), int(axis_points[2][0][1])))
    a3 = np.array((int(axis_points[3][0][0]), int(axis_points[3][0][1])))
    if colors is None:
        if rgb:
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        else:
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    else:
        colors = [colors] * 3

    axes_map = {"x": (a0, colors[0]), "z": (a2, colors[2]), "y": (a1, colors[1])}

    for axis_label, (point, color) in axes_map.items():
        if axis_label in axis:
            if draw_arrow:
                draw_img = cv2.arrowedLine(
                    draw_img, tuple(a3), tuple(point), color, d, tipLength=0.5
                )
            else:
                draw_img = cv2.line(draw_img, tuple(a3), tuple(point), color, d)

    # Add labels for each axis
    if text_label:
        cv2.putText(
            draw_img, "X", tuple(a0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors[0], 3
        )
        cv2.putText(
            draw_img, "Y", tuple(a2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors[1], 3
        )
        cv2.putText(
            draw_img, "Z", tuple(a1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors[2], 3
        )

    if trans:
        # Transparency value
        alpha = 0.50
        # Perform weighted addition of the input image and the overlay
        draw_img = cv2.addWeighted(draw_img, alpha, img, 1 - alpha, 0)

    return draw_img
