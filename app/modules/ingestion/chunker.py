import hashlib
from typing import List, Optional
from app.core.interfaces.ingestion import IChunker
from app.core.schemas.common import ProcessedChunk, ChunkMetadata, ContentType
from app.core.schemas.ingestion import ChunkingConfiguration

class RecursiveCharacterChunker(IChunker):
    """
    Implementation of recursive character splitting strategy.
    Splits text by a list of separators (e.g., \n\n, \n, " ") until 
    chunks are within the configured size.
    """
    
    def __init__(self, config: ChunkingConfiguration):
        self.config = config
        self.separators = ["\n\n", "\n", ". ", "? ", "! ", " "]

    def chunk(self, text: str, metadata: dict) -> List[ProcessedChunk]:
        """
        Splits the input text into ProcessedChunks with preserved metadata.
        """
        raw_chunks = self._split_text(text, self.separators)
        processed_chunks = []
        
        file_name = metadata.get("file_name", "unknown")
        header_1 = metadata.get("header_1")
        header_2 = metadata.get("header_2")
        content_type = metadata.get("content_type", ContentType.TEXT)
        is_table = metadata.get("is_table", False)
        parent_text = metadata.get("parent_text")

        try:
            from tqdm import tqdm
            iterator = tqdm(enumerate(raw_chunks), total=len(raw_chunks), desc=f"📦 Chunking {file_name}", leave=False)
        except ImportError:
            iterator = enumerate(raw_chunks)

        for i, chunk_content in iterator:
            # Generate deterministic ID (will be refined in Phase 3)
            chunk_id = self._generate_id(file_name, chunk_content, i)
            
            meta = ChunkMetadata(
                file_name=file_name,
                chunk_id=chunk_id,
                header_1=header_1,
                header_2=header_2,
                content_type=content_type,
                is_table=is_table,
                parent_text=parent_text
            )
            
            processed_chunks.append(ProcessedChunk(
                content=chunk_content.strip(),
                metadata=meta
            ))
            
        return processed_chunks

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursive splitting implementation."""
        if len(text) <= self.config.chunk_size:
            return [text]
        
        # Find the best separator
        separator = separators[0]
        for s in separators:
            if s in text:
                separator = s
                break
        
        # Split by the separator
        splits = text.split(separator)
        
        # Recombine splits into larger chunks that respect size/overlap
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_len = len(split)
            if current_length + split_len > self.config.chunk_size and current_chunk:
                chunks.append(separator.join(current_chunk))
                # Handle overlap: take last parts of previous chunk
                # For simplicity in this first version, we'll just start fresh
                # but overlap logic should be added here.
                current_chunk = []
                current_length = 0
            
            current_chunk.append(split)
            current_length += split_len + len(separator)
            
        if current_chunk:
            chunks.append(separator.join(current_chunk))
            
        # Recursive check: if any chunk is still too big, split it further
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.config.chunk_size:
                if len(separators) > 1:
                    final_chunks.extend(self._split_text(chunk, separators[1:]))
                else:
                    # Can't split further, just force truncate
                    final_chunks.append(chunk[:self.config.chunk_size])
            else:
                final_chunks.append(chunk)
                
        return final_chunks

    def _generate_id(self, file_name: str, content: str, index: int) -> str:
        """Generates a stable hash-based ID."""
        hash_input = f"{file_name}-{content[:50]}-{index}"
        return hashlib.md5(hash_input.encode()).hexdigest()

if __name__ == "__main__":
    import os
    import json
    from pathlib import Path
    
    # 1. Định nghĩa đường dẫn
    sample_file = "data/sample_data/hybrid/final_super_hybrid_vision.md"
    project_root = Path(__file__).parent.parent.parent.parent.parent
    sample_path = project_root / sample_file
    
    # Đường dẫn file output trong trash
    output_dir = project_root / "trash"
    output_dir.mkdir(exist_ok=True)
    output_json = output_dir / "chunk_test_results.json"

    print(f"🔍 Đang tìm file tại: {sample_path.absolute()}")

    if not sample_path.exists():
        print(f"❌ Lỗi: Không tìm thấy file tại {sample_path}")
    else:
        # 2. Cấu hình Chunking
        config = ChunkingConfiguration(chunk_size=1000, chunk_overlap=200)
        chunker = RecursiveCharacterChunker(config)
        
        # 3. Đọc nội dung
        with open(sample_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # 4. Chuẩn bị metadata
        metadata = {
            "file_name": sample_path.name,
            "header_1": "Lời nói đầu",
            "content_type": ContentType.TEXT
        }
        
        # 5. Thực hiện chunking
        print(f"🚀 Bắt đầu xử lý: {sample_path.name}")
        chunks = chunker.chunk(content, metadata)
        
        # 6. Lưu kết quả sang JSON
        results_data = [chunk.model_dump() for chunk in chunks]
        
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
            
        print(f"\n✅ Hoàn thành! Đã tạo ra {len(chunks)} chunks.")
        print(f"💾 Kết quả đã được lưu tại: {output_json.absolute()}")
        print("="*60)
        
        # Hiển thị demo 1 chunk
        if chunks:
            chunk = chunks[0]
            print(f"DEMO CHUNK #1 | ID: {chunk.metadata.chunk_id}")
            print(f"Nội dung: {chunk.content[:200]}...")
            print("="*60)
