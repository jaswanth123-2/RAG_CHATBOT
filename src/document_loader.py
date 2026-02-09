import os
from typing import List, Dict, Optional
from pathlib import Path
import PyPDF2

class DocumentLoader:
    def __init__(self):
        self.supported_formats = ['.pdf', '.txt', '.docx', '.md']
  
    def load_pdf(self, file_path: str) -> Optional[Dict]:
        text = ""
        metadata = {
            'source': os.path.basename(file_path),
            'type': 'pdf',
            'pages': []
        }
      
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
              
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
                    metadata['pages'].append(page_num + 1)
              
                metadata['num_pages'] = num_pages
                metadata['total_chars'] = len(text)
      
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return None
      
        return {'text': text, 'metadata': metadata}
  
    def load_txt(self, file_path: str) -> Optional[Dict]:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()
          
            metadata = {
                'source': os.path.basename(file_path),
                'type': 'txt',
                'total_chars': len(text)
            }
          
            return {'text': text, 'metadata': metadata}
      
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
  
    def load_markdown(self, file_path: str) -> Optional[Dict]:
        return self.load_txt(file_path)
  
    def load_document(self, file_path: str) -> Optional[Dict]:
        file_ext = Path(file_path).suffix.lower()
      
        if file_ext == '.pdf':
            return self.load_pdf(file_path)
        elif file_ext in ['.txt', '.md']:
            return self.load_txt(file_path)
        else:
            print(f"⚠️ Unsupported format: {file_ext}")
            return None
  
    def load_directory(self, directory_path: str) -> List[Dict]:
        documents = []
      
        if not os.path.exists(directory_path):
            print(f"❌ Directory not found: {directory_path}")
            return documents
      
        print(f"📂 Loading from: {directory_path}")
        print("="*60)
      
        for file_name in sorted(os.listdir(directory_path)):
            file_path = os.path.join(directory_path, file_name)
          
            if os.path.isfile(file_path):
                print(f"📄 {file_name}...", end=" ")
                doc = self.load_document(file_path)
              
                if doc:
                    documents.append(doc)
                    chars = doc['metadata']['total_chars']
                    print(f"✓ {chars:,} chars")
                else:
                    print("✗")
      
        print("="*60)
        print(f"✅ Loaded {len(documents)} documents\n")
        return documents

if __name__ == "__main__":
    loader = DocumentLoader()
    docs = loader.load_directory("data/documents")
  
    print(f"📊 Total: {len(docs)} documents")
    if docs:
        total = sum(d['metadata']['total_chars'] for d in docs)
        print(f"📊 Total characters: {total:,}")