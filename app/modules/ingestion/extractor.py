import os
import re
import time
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, List
import fitz # PyMuPDF
import pymupdf4llm

from app.core.interfaces.ingestion import IVisionAnalyzer, ITableParser, IExtractor

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class HybridExtractor(IExtractor):
    """
    Coordinator class for high-fidelity PDF extraction.
    Combines text, vision analysis (for images), and structured parsing (for tables).
    """

    def __init__(
        self, 
        vision_analyzer: IVisionAnalyzer, 
        table_parser: ITableParser,
        page_num_regex: str = r"^\s*-\s*\d+\s*-\s*$"
    ):
        self.vision_analyzer = vision_analyzer
        self.table_parser = table_parser
        self.page_num_pattern = re.compile(page_num_regex)

    def process_pdf(self, file_path: str, output_dir: str = None) -> str:
        """Main entry point to convert PDF to enriched Markdown."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        logger.info(f"--- Processing document: {path.name} ---")
        
        # Setup image directory if output_dir is provided
        self.img_output_dir = Path(output_dir) / "images" if output_dir else None
        if self.img_output_dir:
            self.img_output_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(path)
        placeholders_data: Dict[str, Dict[str, Any]] = {}
        
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_dir_path = Path(tmp_dir)
                
                # Step 1: Pre-process (Sanitize content)
                self._sanitize_document(doc)
                
                # Step 2: Extract visual and structural elements
                for page in doc:
                    self._process_page_elements(page, doc, tmp_dir_path, placeholders_data)
                
                # Step 3: Text extraction on censored PDF
                redacted_temp_pdf = tmp_dir_path / f"redacted_{path.stem}.pdf"
                doc.save(redacted_temp_pdf)
                
                logger.info("Extracting core text semantics via PyMuPDF4LLM...")
                md_text = pymupdf4llm.to_markdown(str(redacted_temp_pdf))
                
                # Step 4: Reassemble final content
                md_text = self._reassemble_markdown(md_text, placeholders_data)

                with open(Path(output_dir) / "content.md", "w", encoding="utf-8") as f:
                    f.write(md_text)

                return md_text
                
        finally:
            doc.close()

    def _sanitize_document(self, doc: fitz.Document):
        """Erase noisy elements like page numbers."""
        logger.info("Sanitizing document layout (removing page numbers)...")
        for page in doc:
            for b in page.get_text("blocks"):
                if self.page_num_pattern.match(b[4].strip()):
                    page.add_redact_annot(b[:4], fill=(1, 1, 1))
            page.apply_redactions()

    def _process_page_elements(self, page: fitz.Page, doc: fitz.Document, tmp_dir: Path, data: Dict):
        """Analyze page for images and tables, replacing them with placeholders."""
        logger.info(f"  [>] Scanning Page {page.number + 1}...")
        
        # 1. Process Images/Photos
        self._handle_images(page, doc, data)
        
        # 2. Process Tables/Diagrams
        self._handle_structural_elements(page, doc, tmp_dir, data)

    def _handle_images(self, page: fitz.Page, doc: fitz.Document, data: Dict):
        """Detect and analyze images using Vision AI."""
        image_list = page.get_images(full=True)
        for img in image_list:
            xref = img[0]
            rects = page.get_image_rects(xref)
            if not rects: continue
            
            placeholder = f"[IMG_{xref}_P{page.number + 1}]"
            img_info = doc.extract_image(xref)
            img_ext = img_info["ext"]
            img_bytes = img_info["image"]
            
            # Save image if output directory is set
            img_name = f"img_p{page.number + 1}_{xref}.{img_ext}"
            img_path_md = img_name
            if self.img_output_dir:
                with open(self.img_output_dir / img_name, "wb") as f:
                    f.write(img_bytes)
                img_path_md = f"images/{img_name}"
            
            # Get AI Analysis
            ai_content = self._get_vision_summary(img_bytes)
            data[placeholder] = {
                "content": f"![Image]({img_path_md})\n{ai_content}"
            }
            
            # Censor original image and insert placeholder for text extraction
            page.add_redact_annot(rects[0], fill=(1, 1, 1))
            page.apply_redactions()
            page.insert_text(rects[0].tl + (0, 10), placeholder, color=(0, 0, 1))

    def _handle_structural_elements(self, page: fitz.Page, doc: fitz.Document, tmp_dir: Path, data: Dict):
        """Detect tables or complex diagrams and use specialized parsers."""
        tabs_lines = page.find_tables(strategy="lines")
        tabs_text = page.find_tables(horizontal_strategy="text", vertical_strategy="text", snap_tolerance=7)
        
        all_rects = [fitz.Rect(t.bbox) for t in tabs_lines.tables if t.cells]
        all_rects += [fitz.Rect(t.bbox) for t in tabs_text.tables if t.cells]
        merged_rects = self.table_parser.merge_rects(all_rects)

        for i, rect in enumerate(merged_rects):
            if rect.width < 15 or rect.height < 15: continue
            
            placeholder = f"[STRUCT_{page.number + 1}_{i}]"
            extract_rect = rect + (-5, -5, 5, 5) # Add small padding
            
            # Try specialized Table Parser
            table_md = self._extract_table_content(page, doc, extract_rect, tmp_dir, f"struct_p{page.number+1}_{i}.pdf")
            
            if table_md:
                logger.info(f"      [+] Table extracted ({placeholder})")
                data[placeholder] = {"content": table_md}
            else:
                # Fallback to Vision AI for diagrams
                logger.info(f"      [*] Falling back to Vision AI for structure ({placeholder})")
                pix = page.get_pixmap(clip=extract_rect, matrix=fitz.Matrix(2, 2))
                img_bytes = pix.tobytes("png")
                
                # Save diagram image
                diag_name = f"diag_p{page.number+1}_{i}.png"
                img_path_md = diag_name
                if self.img_output_dir:
                    pix.save(self.img_output_dir / diag_name)
                    img_path_md = f"images/{diag_name}"

                ai_content = self._get_vision_summary(img_bytes)
                data[placeholder] = {"content": f"![Diagram]({img_path_md})\n{ai_content}"}

            page.add_redact_annot(extract_rect, fill=(1, 1, 1))
            page.apply_redactions()
            page.insert_text(extract_rect.tl + (0, 10), placeholder, color=(1, 0, 0))


    def _get_vision_summary(self, image_data: bytes, retries: int = 3) -> str:
        """Retry wrapper for vision analysis."""
        for _ in range(retries):
            summary = self.vision_analyzer.summarize_image(image_data)
            if summary: return summary
            time.sleep(1)
        return "[Vision Analysis Failed]"

    def _extract_table_content(self, page: fitz.Page, doc: fitz.Document, rect: fitz.Rect, tmp_dir: Path, name: str) -> str:
        """Extract table from a specific region by creating a snippet PDF."""
        try:
            snippet_path = tmp_dir / name
            new_doc = fitz.open()
            new_page = new_doc.new_page(width=rect.width, height=rect.height)
            new_page.show_pdf_page(new_page.rect, doc, page.number, clip=rect)
            new_doc.save(snippet_path)
            new_doc.close()
            
            md = self.table_parser.extract_table(str(snippet_path))
            # Basic validation: ensure it looks like a markdown table
            return md if len(md.strip()) > 10 and "|" in md else ""
        except Exception as e:
            logger.debug(f"Snippet extraction error: {e}")
            return ""

    def _reassemble_markdown(self, base_text: str, data: Dict[str, Any]) -> str:
        """Replace placeholders with actual analyzed content."""
        logger.info("Compiling final document content...")
        output = base_text
        for placeholder, info in data.items():
            # Ensure we wrap the content with newlines for safe markdown separation
            output = output.replace(placeholder, f"\n\n{info['content']}\n\n")
        
        # Cleanup artifacts from PyMuPDF4LLM
        output = re.sub(r"\*\*==> picture \[.*?\] intentionally omitted <==\*\*", "", output)
        return output

if __name__ == "__main__":
    # Test block for direct execution
    from dotenv import load_dotenv
    load_dotenv()
    
    # 1. Configuration
    test_pdf = "trash/docs/So-tay-Sinh-vien-2025-HVCS-pages-2.pdf"
    output_dir = Path("trash/trash_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Execution
    extractor = HybridExtractor()
    try:
        logger.info(f"Running direct test on: {test_pdf}")
        final_md = extractor.process_pdf(test_pdf, output_dir=output_dir)
        
        # 3. Save result
        output_path = output_dir / "test_refactored_output.md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_md)
            
        logger.info(f"--- Process Complete ---")
        logger.info(f"Result saved to: {output_path}")
    except Exception as e:
        logger.error(f"Test execution failed: {e}", exc_info=True)

