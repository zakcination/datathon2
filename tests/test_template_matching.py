import unittest
import os
from src.template_matching import match_template

class TestTemplateMatching(unittest.TestCase):

    def setUp(self):
        self.input_image_path = "../data/input_images/apartment_plan.png"
        self.template_path = "../data/templates/hall_table.png"
        self.non_existent_path = "../data/templates/non_existent.png"

    def test_template_matching_output(self):
        matches = match_template(self.input_image_path, self.template_path)
        self.assertIsInstance(matches, list)
        self.assertGreaterEqual(len(matches), 0, "Should return a list of match locations.")

    def test_template_not_found(self):
        with self.assertRaises(FileNotFoundError):
            match_template(self.non_existent_path, self.template_path)

    def test_invalid_threshold(self):
        matches = match_template(self.input_image_path, self.template_path, threshold=1.5)
        self.assertEqual(matches, [], "Should return an empty list if threshold is too high.")

if __name__ == '__main__':
    unittest.main()
