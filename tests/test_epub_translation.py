import unittest
import os
from src.document_translator import DocumentTranslator
from src.inference import TranslationInference
from src.utils import load_config

class TestEPUBTranslation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.engine = TranslationInference(cls.config)
        cls.translator = DocumentTranslator(cls.engine)
        # Create a minimal test EPUB file
        cls.test_epub_path = 'tests/test_sample.epub'
        cls.translated_epub_path = 'tests/test_sample_translated.epub'
        if not os.path.exists(cls.test_epub_path):
            from ebooklib import epub
            book = epub.EpubBook()
            book.set_identifier('id123456')
            book.set_title('Test Book')
            book.set_language('ja')
            c1 = epub.EpubHtml(title='Intro', file_name='intro.xhtml', lang='ja')
            c1.content = '<h1>こんにちは</h1><p>これはテストです。</p>'
            book.add_item(c1)
            book.spine = ['nav', c1]
            book.add_item(epub.EpubNcx())
            book.add_item(epub.EpubNav())
            epub.write_epub(cls.test_epub_path, book)

    def test_epub_translation(self):
        # Translate the test EPUB from Japanese to English
        result_path = self.translator.translate_document(
            self.test_epub_path,
            self.translated_epub_path,
            progress_callback=None,
            direction='ja-en'
        )
        self.assertTrue(os.path.exists(result_path))
        # Check that the translated file contains English text
        from ebooklib import epub
        book = epub.read_epub(result_path)
        found_english = False
        for item in book.get_items():
            if item.get_type() == 9:  # ITEM_DOCUMENT
                content = item.get_content().decode('utf-8')
                if any(word in content.lower() for word in ['test', 'hello']):
                    found_english = True
        self.assertTrue(found_english)

if __name__ == '__main__':
    unittest.main()
