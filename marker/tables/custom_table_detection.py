from collections import defaultdict
from typing import List
import pandas as pd
from marker.schema.page import Page
from marker.tables.intersections import (
    detect_rowwise_intersection,
    detect_boxes,
    get_cells,
    fill_text_in_cells,
)
from marker.tables.utils import (
    sort_table_blocks,
    replace_dots,
    replace_newlines,
    normalize_bbox,
    denormalize_bbox,
)
from marker.tables.detections import (
    cluster_horizontal_lines,
    cluster_vertical_lines,
    detect_borderlines,
    detect_horizontal_textlines,
    extend_lines,
    filter_non_intersecting_lines,
)
from marker.schema.block import Line, Span, Block
from marker.tables.schema import Rectangle
from marker.tables.utils import save_table_image, remove_extra_blocks
from marker.settings import settings
import tempfile
import numpy as np
import pytesseract
import fitz
import torch
import cv2
import PIL


def ocr_img(image: PIL.Image):
    tesseract_config = "-l hin --oem 3 --psm 6"
    data = pytesseract.image_to_data(
        image, config=tesseract_config, output_type=pytesseract.Output.DICT
    )
    return data


def get_page(pdf, page_num):
    # TODO: Add page check if page_num is valid
    page = pdf.load_page(page_num)
    pix = page.get_pixmap(dpi=180)
    image = PIL.Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return image


def extract_table(pdf: fitz.Document, page_num: int, encoder, detector):
    image = get_page(pdf, page_num)

    encodings = encoder(image, return_tensors="pt")

    with torch.no_grad():
        outputs = detector(**encodings)
    width, height = image.size
    target_size = [(height, width)]
    results = encoder.post_process_object_detection(
        outputs,
        threshold=settings.TABLE_TRANSFORMER_DETECTION_THRESHOLD,
        target_sizes=target_size,
    )[0]

    table_extraction_results = []
    for idx, ex_box in enumerate(results["boxes"]):
        box = Rectangle.fromCoords(*ex_box)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            image_path = temp_file.name
            save_table_image(image, box, image_path)
            image = PIL.Image.open(image_path)
            img_cv2 = cv2.imread(image_path)
            ocr_data = ocr_img(image)

            horizontal_lines_bbox, vertical_line_bboxes = detect_borderlines(image_path)
            text_line_bbox, text_line_height = detect_horizontal_textlines(
                ocr_data, image
            )

            text_line_bbox = [l for l in text_line_bbox if l.width > l.height]

            _, _, _, new_horizontal_lines_bbox = filter_non_intersecting_lines(
                text_line_bbox,
                horizontal_lines_bbox,
                horizontal_lines_height=text_line_height / 4,
            )

            horizontal_lines_bbox_clustered = cluster_horizontal_lines(
                new_horizontal_lines_bbox, text_line_height
            )
            vertical_line_bboxes_clustered = cluster_vertical_lines(
                vertical_line_bboxes, text_line_height
            )

            horizontal_lines_bbox_clustered_filt, _, _, _ = (
                filter_non_intersecting_lines(
                    horizontal_lines_bbox_clustered,
                    vertical_line_bboxes_clustered,
                    vertical_lines_width=text_line_height / 8,
                )
            )

            h_lines, v_lines = extend_lines(
                img_cv2,
                horizontal_lines_bbox_clustered_filt,
                vertical_line_bboxes_clustered,
                text_line_height / 8,
            )
            # for h in h_lines:
            #     h.draw(img_cv2)

            # for v in v_lines:
            #     v.draw(img_cv2)

            # cv2.imwrite(f"out/pg_{page_num}.png", img_cv2)
            table_extraction_results.append(
                (h_lines, v_lines, ocr_data, ex_box, image_path)
            )
    return table_extraction_results


def table_detection(doc, page, pnum: int, table_detect_model, feature_extractor):
    total_tables = 0
    extraction_results = extract_table(doc, pnum, feature_extractor, table_detect_model)
    for h_lines, v_lines, ocr_data, table_bbox, table_img_path in extraction_results:
        img = cv2.imread(table_img_path)
        pg_img = np.array(get_page(doc, pnum))
        remove_extra_blocks(page, table_bbox, pg_img)
        cv2.imwrite(f"output/{pnum}.png", pg_img)
        print("H Lines: ", len(h_lines))
        print("H Lines: ", len(v_lines))

        h_lines.sort(key=lambda l: l.y)
        v_lines.sort(key=lambda l: l.x)
        rowwise_intersections = detect_rowwise_intersection(h_lines, v_lines)
        for p_r in rowwise_intersections:
            for p in p_r:
                p.draw(img)
        boxes = detect_boxes(rowwise_intersections)
        cells = get_cells(boxes, h_lines, v_lines)

        words_original = [
            {"x": x, "y": y, "width": width, "height": height, "text": text}
            for x, y, width, height, text in zip(
                ocr_data["left"],
                ocr_data["top"],
                ocr_data["width"],
                ocr_data["height"],
                ocr_data["text"],
            )
            if text.strip()
        ]
        fill_text_in_cells(words_original, cells, img)

        # row wise cell segregation
        rows = defaultdict(dict)
        for cell in cells:
            rows[cell.r][cell.c] = cell.text

        table_df = pd.DataFrame.from_dict(rows, orient="index")
        if table_df.empty:
            print("The DataFrame is empty")
            continue
        table_df = table_df.drop_duplicates(keep="last")
        table_df.columns = table_df.iloc[0]
        table_df = table_df[1:].reset_index(drop=True)
        # table_df.to_csv(f"{pnum}_table.csv", index=False)
        md = table_df.to_markdown(index=False)

        table_block = Block(
            bbox=table_bbox,
            block_type="Table",
            pnum=pnum,
            lines=[
                Line(
                    bbox=table_bbox,
                    spans=[
                        Span(
                            bbox=table_bbox,
                            span_id=f"{pnum}_table",
                            font="Table",
                            font_size=0,
                            font_weight=0,
                            block_type="TABLE",
                            text=md,
                        )
                    ],
                )
            ],
        )
        page.blocks.append(table_block)
        total_tables += 1
        # cv2.imwrite(f"{i_pg}.png", img)
    return total_tables
