import unittest
from src.inference import TranslationInference
from src.utils import load_config

class TestTranslationInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.engine = TranslationInference(cls.config)

    def test_en_zh(self):
        result = self.engine.translate('hello', 'en-zh')
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_zh_en(self):
        result = self.engine.translate('你好', 'zh-en')
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_en_ja(self):
        result = self.engine.translate('hello', 'en-ja')
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_ja_en(self):
        result = self.engine.translate('こんにちは', 'ja-en')
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

if __name__ == '__main__':
    unittest.main()
