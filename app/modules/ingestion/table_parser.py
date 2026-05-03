from app.core.interfaces.ingestion import ITableParser
from docling.document_converter import DocumentConverter

class DoclingTableParser(ITableParser):
    """Implementation of table parsing using Docling."""
    
    def __init__(self):
        # Lazy initialization to avoid slow imports/startup overhead unless used
        self._converter = None

    @property
    def converter(self):
        if self._converter is None:
            self._converter = DocumentConverter()
        return self._converter

    def extract_table(self, file_path: str) -> str:
        """Extracts table via Docling and exports to markdown."""
        result = self.converter.convert(file_path)
        md = result.document.export_to_markdown()
        return md

    @staticmethod
    def merge_rects(rect_list, tolerance=10):
        """Utility to merge close rects to ensure we don't break multi-part tables."""
        if not rect_list: return []
        rect_list.sort(key=lambda r: (r.y0, r.x0))
        merged = [rect_list[0]]
        for current in rect_list[1:]:
            prev = merged[-1]
            if current.intersects(prev) or (current.y0 - prev.y1 < tolerance):
                merged[-1] = prev | current
            else:
                merged.append(current)
        return merged
