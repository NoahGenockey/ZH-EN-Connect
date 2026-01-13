"""
LinguaBridge Local - Document Translation Module
Supports PDF and EPUB book/document translation with structure preservation.
"""

import os
import logging
from typing import Optional, Dict, List, Tuple
from pathlib import Path

# PDF libraries
try:
    import PyPDF2
    import pdfplumber
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# EPUB library
try:
    from ebooklib import epub
    from bs4 import BeautifulSoup
    EPUB_AVAILABLE = True
except ImportError:
    EPUB_AVAILABLE = False


# Always use absolute imports for compatibility with test runner
from src.inference import TranslationInference
from src.utils import load_config


class DocumentTranslator:
    """
    Translates entire documents (PDF, EPUB) while preserving structure.
    """
    
    def __init__(self, inference_engine: TranslationInference):
        """
        Initialize document translator.
        
        Args:
            inference_engine: Translation inference engine
        """
        self.inference = inference_engine
        self.logger = logging.getLogger('LinguaBridge.DocumentTranslator')
        
        # Register Chinese font for PDF output
        self._register_chinese_font()
    
    def _register_chinese_font(self):
        """Register Chinese font for PDF generation."""
        try:
            # Try to use system fonts
            # Windows: SimSun, SimHei
            # macOS: PingFang, STHeiti
            # Linux: WenQuanYi, Noto Sans CJK
            
            font_paths = [
                # Windows
                r'C:\Windows\Fonts\simsun.ttc',
                r'C:\Windows\Fonts\msyh.ttc',
                # macOS
                '/System/Library/Fonts/PingFang.ttc',
                '/Library/Fonts/Arial Unicode.ttf',
                # Linux
                '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        pdfmetrics.registerFont(TTFont('Chinese', font_path))
                        self.logger.info(f"Registered Chinese font: {font_path}")
                        return
                    except:
                        continue
            
            self.logger.warning("No Chinese font found. PDF output may not display Chinese correctly.")
        except Exception as e:
            self.logger.warning(f"Font registration failed: {e}")
    
    def translate_document(self, input_path: str, output_path: Optional[str] = None,
                          progress_callback=None, direction: str = 'en-zh') -> str:
        """
        Translate a document file.
        
        Args:
            input_path: Path to input document
            output_path: Path to output document (auto-generated if None)
            progress_callback: Function to call with progress updates (0-100)
            direction: Translation direction ('en-zh' or 'zh-en')
        
        Returns:
            Path to translated document
        """
        input_path = Path(input_path)
        ext = input_path.suffix.lower()
        
        if not output_path:
            output_path = input_path.parent / f"{input_path.stem}_translated{ext}"
        
        self.logger.info(f"Translating {ext} document: {input_path} ({direction})")
        
        if ext == '.pdf':
            return self._translate_pdf(str(input_path), str(output_path), progress_callback, direction)
        elif ext == '.epub':
            return self._translate_epub(str(input_path), str(output_path), progress_callback, direction)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def _translate_pdf(self, input_path: str, output_path: str, 
                       progress_callback=None, direction: str = 'en-zh') -> str:
        """Translate PDF document."""
        if not PDF_AVAILABLE:
            raise ImportError("PDF libraries not installed. Run: pip install PyPDF2 pdfplumber reportlab")
        
        # Extract text from PDF
        self.logger.info("Extracting text from PDF...")
        texts = []
        
        with pdfplumber.open(input_path) as pdf:
            total_pages = len(pdf.pages)
            
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    texts.append(text)
                
                if progress_callback:
                    progress = int((i + 1) / total_pages * 30)  # 0-30%
                    progress_callback(progress, f"Extracting page {i+1}/{total_pages}")
        
        # Translate texts using batch processing
        self.logger.info(f"Translating {len(texts)} pages...")
        translated_texts = []
        
        for i, text in enumerate(texts):
            try:
                # Split into paragraphs for better translation
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                # Translate all paragraphs, regardless of length
                translated_paragraphs = []
                if paragraphs:
                    translated = self.inference.batch_translate(paragraphs, batch_size=8, direction=direction)
                    translated_paragraphs = translated
                translated_texts.append('\n\n'.join(translated_paragraphs))
            except Exception as e:
                self.logger.error(f"Error translating page {i+1}: {e}")
                translated_texts.append(f"[Translation Error on page {i+1}: {e}]")
            if progress_callback:
                progress = 30 + int((i + 1) / len(texts) * 60)  # 30-90%
                progress_callback(progress, f"Translating page {i+1}/{len(texts)}")
        
        # Generate translated PDF
        self.logger.info("Generating translated PDF...")
        self._create_pdf(translated_texts, output_path)
        
        if progress_callback:
            progress_callback(100, "Translation complete!")
        
        self.logger.info(f"Translated PDF saved to: {output_path}")
        return output_path
    
    def _create_pdf(self, texts: List[str], output_path: str):
        """Create PDF from translated texts."""
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Create Chinese-compatible style
        chinese_style = styles['Normal'].clone('ChineseStyle')
        try:
            chinese_style.fontName = 'Chinese'
        except:
            pass  # Fallback to default font
        chinese_style.fontSize = 12
        chinese_style.leading = 18
        
        story = []
        
        for text in texts:
            # Add paragraph
            para = Paragraph(text.replace('\n', '<br/>'), chinese_style)
            story.append(para)
            story.append(Spacer(1, 12))
        
        doc.build(story)
    
    def _translate_epub(self, input_path: str, output_path: str,
                        progress_callback=None, direction: str = 'en-zh') -> str:
        """Translate EPUB document."""
        if not EPUB_AVAILABLE:
            raise ImportError("EPUB libraries not installed. Run: pip install ebooklib beautifulsoup4")
        
        # Read EPUB
        self.logger.info("Reading EPUB...")
        book = epub.read_epub(input_path)
        
        # Create new EPUB for translation
        translated_book = epub.EpubBook()
        translated_book.set_identifier(book.get_metadata('DC', 'identifier')[0][0] + '_zh')
        translated_book.set_title(book.get_metadata('DC', 'title')[0][0] + ' (中文翻译)')
        translated_book.set_language('zh')
        
        # Copy metadata
        authors = book.get_metadata('DC', 'creator')
        if authors:
            for author in authors:
                translated_book.add_author(author[0])
        
        # Translate chapters
        items = list(book.get_items())
        translated_items = []
        
        for i, item in enumerate(items):
            if item.get_type() == 9:  # ITEM_DOCUMENT
                try:
                    self.logger.info(f"Translating chapter {i+1}/{len(items)}...")
                    # Parse HTML
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    # Collect all text nodes to translate (no length filter)
                    text_nodes = []
                    text_contents = []
                    for text_node in soup.find_all(string=True):
                        if text_node.parent.name not in ['script', 'style']:
                            text = text_node.strip()
                            if text:
                                text_nodes.append(text_node)
                                text_contents.append(text)
                    # Batch translate all text at once (much faster!)
                    if text_contents:
                        translated_texts = self.inference.batch_translate(text_contents, batch_size=8, direction=direction)
                        # Replace text nodes with translations
                        for text_node, translated in zip(text_nodes, translated_texts):
                            text_node.replace_with(translated)
                    # Create translated chapter
                    translated_item = epub.EpubHtml(
                        title=item.get_name(),
                        file_name=item.get_name(),
                        lang='zh'
                    )
                    translated_item.content = str(soup).encode('utf-8')
                    translated_book.add_item(translated_item)
                    translated_items.append(translated_item)
                except Exception as e:
                    self.logger.error(f"Error translating chapter {i+1}: {e}")
                if progress_callback:
                    progress = int((i + 1) / len(items) * 100)
                    progress_callback(progress, f"Translating chapter {i+1}/{len(items)}")
            else:
                # Copy non-document items (images, CSS, etc.)
                translated_book.add_item(item)
        
        # Set spine and TOC
        translated_book.spine = ['nav'] + translated_items
        translated_book.add_item(epub.EpubNcx())
        translated_book.add_item(epub.EpubNav())
        
        # Write EPUB
        epub.write_epub(output_path, translated_book)
        
        if progress_callback:
            progress_callback(100, "Translation complete!")
        
        self.logger.info(f"Translated EPUB saved to: {output_path}")
        return output_path


def check_dependencies() -> Dict[str, bool]:
    """Check if document translation dependencies are installed."""
    return {
        'pdf': PDF_AVAILABLE,
        'epub': EPUB_AVAILABLE
    }


def install_dependencies():
    """Print installation instructions for missing dependencies."""
    deps = check_dependencies()
    
    if not all(deps.values()):
        print("\n" + "="*60)
        print("Document Translation - Missing Dependencies")
        print("="*60)
        
        if not deps['pdf']:
            print("\n📄 PDF Support:")
            print("   pip install PyPDF2 pdfplumber reportlab")
        
        if not deps['epub']:
            print("\n📚 EPUB Support:")
            print("   pip install ebooklib beautifulsoup4 lxml")
        
        print("\n" + "="*60)


if __name__ == "__main__":
    # Dependency check
    deps = check_dependencies()
    print("\nDocument Translation Support:")
    print(f"  PDF:  {'✅ Available' if deps['pdf'] else '❌ Not installed'}")
    print(f"  EPUB: {'✅ Available' if deps['epub'] else '❌ Not installed'}")
    
    if not all(deps.values()):
        install_dependencies()
