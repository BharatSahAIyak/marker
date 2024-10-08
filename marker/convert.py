import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning
)  # Filter torch pytree user warnings

from marker.tables.table import table_detection
import pypdfium2 as pdfium  # Needs to be at the top to avoid warnings
from PIL import Image

from marker.utils import flush_cuda_memory
from marker.debug.data import dump_bbox_debug_data
from marker.layout.layout import surya_layout, annotate_block_types
from marker.layout.order import surya_order, sort_blocks_in_reading_order
from marker.ocr.lang import replace_langs_with_codes, validate_langs
from marker.ocr.detection import surya_detection
from marker.ocr.recognition import run_ocr
from marker.pdf.extract_text import get_text_blocks
from marker.cleaners.headers import filter_header_footer, filter_common_titles
from marker.equations.equations import replace_equations
from marker.pdf.utils import find_filetype
from marker.postprocessors.editor import edit_full_text
from marker.cleaners.code import identify_code_blocks, indent_blocks
from marker.cleaners.bullets import replace_bullets
from marker.cleaners.headings import split_heading_blocks
from marker.cleaners.fontstyle import find_bold_italic
from marker.postprocessors.markdown import merge_spans, merge_lines, get_full_text
from marker.cleaners.text import cleanup_text
from marker.images.extract import extract_images
from marker.images.save import images_to_dict
from marker.ocr.langdetect import get_text, detect_language_text, detect_language_ocr, keep_most_frequent_element

from typing import List, Dict, Tuple, Optional
from marker.settings import settings
def convert_single_pdf(
    fname: str,
    model_lst: List,
    max_pages: int = None,
    start_page: int = None,
    metadata: Optional[Dict] = None,
    langs: Optional[List[str]] = None,
    batch_multiplier: int = 1,
) -> Tuple[str, Dict[str, Image.Image], Dict]:
    # Set language needed for OCR
    if langs is None:
        langs = [settings.DEFAULT_LANG]

    OCR_ALL_PAGES = False

    # if metadata:
    #     langs = metadata.get("languages", langs)
    #     if metadata.get("OCR_ALL_PAGES"):
    #         OCR_ALL_PAGES = True

    # langs = replace_langs_with_codes(langs)
    # validate_langs(langs)

    # Find the filetype
    filetype = find_filetype(fname)

    # Setup output metadata
    out_meta = {
        "filetype": filetype,
    }

    if filetype == "other":  # We can't process this file
        return "", {}, out_meta

    # Get initial text blocks from the pdf
    doc = pdfium.PdfDocument(fname)
    pages, toc = get_text_blocks(doc, fname, max_pages=max_pages, start_page=start_page)
    out_meta.update(
        {
            "toc": toc,
            "pages": len(pages),
        }
    )

    valid_langs=["en","hi","or"]

    # Detecting language of the text layer present. Getting empty means OCR is needed.
    language = detect_language_text(get_text(pages))
    langs = [language]
    # validate_langs(langs)

    print("langs >",langs)
    if language not in valid_langs:
        OCR_ALL_PAGES = True
        language = detect_language_ocr(fname)
        langs = language
        # if language in valid_langs:
        #     pages = convert_pages_to_unicode(pages)

        # else:
        if keep_most_frequent_element(language)[0] not in valid_langs: 
            langs = ["en"]
        langs=list(set(langs))
        if "unknown" in langs:
            langs.remove("unknown")
        for lang in langs:
            if lang not in valid_langs:
                langs.remove(lang)
        if len(langs)==0:
            langs = ["en"]
        langs=list(langs)

    print("langs >",langs)
    

    #     OCR_ALL_PAGES=True
    #     language = detect_language_ocr(fname)
    #     langs = language
    #     print("langs >",langs)

    # Trim pages from doc to align with start page
    if start_page:
        for page_idx in range(start_page):
            doc.del_page(0)

    # Unpack models from list
    texify_model, layout_model, order_model, edit_model, detection_model, ocr_model = (
        model_lst
    )

    # Identify text lines on pages
    surya_detection(doc, pages, detection_model, batch_multiplier=batch_multiplier)
    flush_cuda_memory()

    # OCR pages as needed
    pages, ocr_stats = run_ocr(
        doc, pages, langs, ocr_model, OCR_ALL_PAGES, batch_multiplier=batch_multiplier
    )
    flush_cuda_memory()

    out_meta["ocr_stats"] = ocr_stats
    if len([b for p in pages for b in p.blocks]) == 0:
        print(f"Could not extract any text blocks for {fname}")
        return "", {}, out_meta

    surya_layout(doc, pages, layout_model, batch_multiplier=batch_multiplier)
    flush_cuda_memory()

    # Find headers and footers
    bad_span_ids = filter_header_footer(pages)
    out_meta["block_stats"] = {"header_footer": len(bad_span_ids)}

    # Add block types in
    annotate_block_types(pages)

    # Dump debug data if flags are set
    dump_bbox_debug_data(doc, fname, pages)
    table_detection(fname, pages, max_pages=max_pages)
    # Find reading order for blocks
    # Sort blocks by reading order
    surya_order(doc, pages, order_model, batch_multiplier=batch_multiplier)
    sort_blocks_in_reading_order(pages)
    flush_cuda_memory()

    # Fix code blocks
    code_block_count = identify_code_blocks(pages)
    out_meta["block_stats"]["code"] = code_block_count
    indent_blocks(pages)

    # Fix table blocks
    # table_count = format_tables(pages)
    # out_meta["block_stats"]["table"] = table_count

    from marker.schema.block import Span, Line, Block

    for page_num, page in enumerate(pages):
        for block in page.blocks:
            block.filter_spans(bad_span_ids)
            block.filter_bad_span_types()

        page_number_span = Span(
            bbox=[10, 10, 10, 10],
            text=f"<header> PAGE NUMBER {page_num + 1} </header>\n",
            span_id="0_0",
            font="Times-New-Roman_bold",
            font_weight=0.0,
            font_size=0.0,
        )

        page_number_line = Line(bbox=[10, 10, 10, 10], spans=[page_number_span])

        page_number_block = Block(
            bbox=[10, 10, 10, 10],
            lines=[page_number_line],
            pnum=page_num + 1,
            block_type="PAGE_NUMBER",
        )

        page.blocks.insert(0, page_number_block)

    filtered, eq_stats = replace_equations(
        doc, pages, texify_model, batch_multiplier=batch_multiplier
    )
    flush_cuda_memory()
    out_meta["block_stats"]["equations"] = eq_stats

    # Extract images and figures
    if settings.EXTRACT_IMAGES:
        extract_images(doc, pages)

    # Split out headers
    split_heading_blocks(pages)
    find_bold_italic(pages)

    # Copy to avoid changing original data
    merged_lines = merge_spans(filtered)
    text_blocks = merge_lines(merged_lines)
    text_blocks = filter_common_titles(text_blocks)
    full_text = get_full_text(text_blocks)

    # Handle empty blocks being joined
    full_text = cleanup_text(full_text)

    # Replace bullet characters with a -
    full_text = replace_bullets(full_text)

    # Postprocess text with editor model
    full_text, edit_stats = edit_full_text(
        full_text, edit_model, batch_multiplier=batch_multiplier
    )
    flush_cuda_memory()
    out_meta["postprocess_stats"] = {"edit": edit_stats}
    doc_images = images_to_dict(pages)

    language = detect_language_text(full_text)
    langs=[language]
    print("langs >",langs)
    out_meta.update(
        {
            "languages": langs
        }
    )

    return full_text, doc_images, out_meta, merged_lines
