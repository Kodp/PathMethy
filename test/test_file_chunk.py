import unittest
import tempfile
import os, sys
sys.path.append(os.path.abspath('../utils'))
from file_chunk import split_file, combine_files  # 确保正确导入这些函数

class TestFileSplitCombine(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_file_path = os.path.join(self.temp_dir.name, 'test_file.txt')
        with open(self.test_file_path, 'wb') as f:
            f.write(os.urandom(100000))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_split_and_combine(self):
        parts = split_file(self.test_file_path, 25000)
        self.assertTrue(len(parts) > 1)

        combined_file_path = os.path.join(self.temp_dir.name, 'combined_file.txt')
        combine_files(parts, combined_file_path)

        with open(self.test_file_path, 'rb') as original, open(combined_file_path, 'rb') as combined:
            self.assertEqual(original.read(), combined.read())

if __name__ == '__main__':
    unittest.main()
