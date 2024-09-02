import re
from typing import List
from marker.schema.page import Page
import PIL.Image
from marker.tables.schema import Rectangle

def sort_table_blocks(blocks, tolerance=5):
    vertical_groups = {}
    for block in blocks:
        if hasattr(block, "bbox"):
            bbox = block.bbox
        else:
            bbox = block["bbox"]
        group_key = round(bbox[1] / tolerance)
        if group_key not in vertical_groups:
            vertical_groups[group_key] = []
        vertical_groups[group_key].append(block)

    # Sort each group horizontally and flatten the groups into a single list
    sorted_blocks = []
    for _, group in sorted(vertical_groups.items()):
        sorted_group = sorted(
            group, key=lambda x: x.bbox[0] if hasattr(x, "bbox") else x["bbox"][0]
        )
        sorted_blocks.extend(sorted_group)

    return sorted_blocks


def replace_dots(text):
    dot_pattern = re.compile(r"(\s*\.\s*){4,}")
    dot_multiline_pattern = re.compile(r".*(\s*\.\s*){4,}.*", re.DOTALL)

    if dot_multiline_pattern.match(text):
        text = dot_pattern.sub(" ", text)
    return text


def replace_newlines(text):
    # Replace all newlines
    newline_pattern = re.compile(r"[\r\n]+")
    return newline_pattern.sub(" ", text.strip())



def save_table_image(page_img: PIL.Image, box: Rectangle, output_path: str, padding=30):
    """Convert the tensor to a list of Python floats and then to integers"""
    padding = 30

    # box = [int(coord) for coord in box.tolist()]

    # Crop format: (left, upper, right, lower)
    # left, upper, right, lower = box
    cropped_image = page_img.crop(
        (
            box.left - padding,
            box.top - padding,
            box.right + padding,
            box.bottom + padding,
        )
    )
    cropped_image.save(output_path)
    return output_path

def normalize_bbox(bbox, width, height):
    x_min, y_min, x_max, y_max = bbox
    
    x_min_norm = x_min / width
    y_min_norm = y_min / height
    x_max_norm = x_max / width
    y_max_norm = y_max / height
    
    return [x_min_norm, y_min_norm, x_max_norm, y_max_norm]

def denormalize_bbox(normalized_bbox, width, height):
    x_min_norm, y_min_norm, x_max_norm, y_max_norm = normalized_bbox
    
    x_min = x_min_norm * width
    y_min = y_min_norm * height
    x_max = x_max_norm * width
    y_max = y_max_norm * height
    
    # Round to integers since pixel coordinates are typically whole numbers
    return [round(x_min), round(y_min), round(x_max), round(y_max)]



def remove_extra_blocks(page: Page, table_bbox: list, img, ):
    height, width = img.shape[:2]
    print(table_bbox)
    box = Rectangle.fromCoords(*table_bbox)
    box.draw(img)
    blocks_to_remove = []
    for blk in page.blocks:
        bbox_norm =  normalize_bbox(blk.bbox, page.width,page.height)
        bbox = denormalize_bbox(bbox_norm,width,height )
    
        block_rec = Rectangle.fromCoords(*bbox)
        if (box.top <= block_rec.top <= box.bottom) or (
            box.top <= block_rec.bottom <= box.bottom
        ):
            blocks_to_remove.append(blk)
        else:
            block_rec.draw(img)
    print(len(page.blocks))
    for blk in blocks_to_remove:
        page.blocks.remove(blk)
    print(len(page.blocks))
