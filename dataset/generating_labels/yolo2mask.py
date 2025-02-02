import os
from PIL import Image, ImageDraw
import numpy as np


def convert_yolo_to_mask(image_path, annotation_path, single_target=True):
    """
    Convert a YOLO annotation to a mask for a given image.

    :param image_path: Path to the image file.
    :param annotation_path: Path to the YOLO annotation file (.txt).
    :param single_target: Boolean indicating whether there is only one target class.
                          If True, the mask will have 255 for target regions.
                          If False, the mask will use label_id + 1 for target regions.
    :return: The generated mask as a PIL Image.
    """

    image = Image.open(image_path)
    width, height = image.size

    def load_yolo_annotation(file_path):
        """Load YOLO format annotation file."""
        with open(file_path, 'r') as file:
            lines = file.readlines()

        annotations = []
        for line in lines:
            parts = line.strip().split()
            label_id = int(parts[0])
            points = [(float(parts[i]), float(parts[i + 1])) for i in range(1, len(parts), 2)]
            annotations.append((label_id, points))

        return annotations

    def draw_polygon_on_mask(draw, points, value, image_size):
        """Draw polygon on mask with specified value."""
        width, height = image_size
        abs_points = [(int(x * width), int(y * height)) for x, y in points]
        draw.polygon(abs_points, fill=value)

    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    annotation = load_yolo_annotation(annotation_path)

    for label_id, points in annotation:
        if single_target:
            draw_polygon_on_mask(draw, points, 1, (width, height))
        else:
            draw_polygon_on_mask(draw, points, label_id + 1, (width, height))

    return mask


# Example usage
if __name__ == '__main__':
    image_path = r"D:\Code\pycode\dataset_all\tf_version3\images\val\val5.jpg"
    annotation_path = r"D:\Code\pycode\dataset_all\tf_version3\labels\segmentation\val\val5.txt"

    mask = convert_yolo_to_mask(image_path, annotation_path, single_target=True)

    mask.show()
    # mask.save(r'D:\Code\pycode\dataset_all\tf_version3\images\mask\val5_.png')

# if __name__ == '__main__':
#     image_shape = (3632, 2760)
#     annotation_path = r"D:\Code\pycode\dataset_all\tf_version3\labels\segmentation\val\val5.txt"
#
#     mask = convert_yolo_to_mask(image_shape, annotation_path, single_target=True)
#
#     mask.show()
#     mask.save(r'D:\Code\pycode\dataset_all\tf_version3\images\mask\val5_.png')
