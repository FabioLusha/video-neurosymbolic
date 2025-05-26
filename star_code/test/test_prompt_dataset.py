import json
import os, sys
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
import pytest

import requests

# Add src directory to path FIRST (before other imports)
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)  # Insert at start to prioritize local imports

import prompt_formatters as pf
from datasets import PromptDataset


class TestPromptDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create temporary test files
        cls.temp_dir = tempfile.TemporaryDirectory()

        # Create sample QA file
        cls.qa_file = os.path.join(cls.temp_dir.name, "qa.json")
        cls.qa_data = [
            {"question_id": "1", "question": "q1", "stsg": "person1"},
            {"question_id": "5", "question": "q2", "stsg": "person2"},
            {"question_id": "7", "question": "q3", "stsg": "person3"},
        ]
        with open(cls.qa_file, "w") as f:
            json.dump(cls.qa_data, f)

        # Create sample STSG file
        cls.stsg_file = os.path.join(cls.temp_dir.name, "stsg.jsonl")
        cls.stsg_data = [
            {"question_id": "1", "stsg": "object1"},
            {"question_id": "5", "stsg": "object2"},
            {"question_id": "7", "stsg": "object3"},
        ]
        with open(cls.stsg_file, "w") as f:
            for item in cls.stsg_data:
                f.write(json.dumps(item) + "\n")

        # Simple prompt formatter for testing
        cls.prompt_formatter = pf.OpenEndedPrompt("Q: {question} STSG: {stsg}")

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_init_with_stsg(self):
        """Test initialization with STSG file"""
        dataset = PromptDataset(self.qa_file, self.prompt_formatter, self.stsg_file)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(
            self.prompt_formatter.format({**self.qa_data[0], **self.stsg_data[0]}),
            "Q: q1 STSG: object1",
        )
    @pytest.mark.skip(reason="The functionality is deprecated in the new version") 
    def test_getitem_with_stsg(self):
        """Test __getitem__ with STSG data"""
        dataset = PromptDataset(
            self.qa_file, self.prompt_formatter, stsg_file_path=self.stsg_file
        )
        sample = dataset[0]
        self.assertEqual("Q: q1 STSG: object1", sample["prompt"])
        # Should have merged STSG data
        self.assertEqual("1", sample["qid"])

    def test_default_stsg(self):
        """Test __getitem__ with STSG data"""
        dataset = PromptDataset(self.qa_file, self.prompt_formatter)
        self.assertEqual("Q: q1 STSG: person1", dataset[0]["prompt"])
        self.assertEqual("1", dataset[0]["qid"])

        self.assertEqual("Q: q2 STSG: person2", dataset[1]["prompt"])
        self.assertEqual("5", dataset[1]["qid"])

        self.assertEqual("Q: q3 STSG: person3", dataset[2]["prompt"])
        self.assertEqual("7", dataset[2]["qid"])

    def test_filter_by_ids(self):
        """Test filtering by specific IDs"""
        dataset = PromptDataset(
            self.qa_file, self.prompt_formatter, self.stsg_file, ids=["1", "5"]
        )
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]["qid"], "1")
        self.assertEqual(dataset[1]["qid"], "5")

    def test_qid_key_detection(self):
        """Test automatic detection of question ID key"""
        dataset1 = PromptDataset(self.qa_file, self.prompt_formatter)
        self.assertEqual(dataset1.q_id_key, "question_id")  # First item has 'qid'

        # Test with different data
        modified_data = [{"question_id": "1", "question": "Test"}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
            json.dump(modified_data, f)
            f.flush()
            dataset2 = PromptDataset(f.name, self.prompt_formatter)
            self.assertEqual(dataset2.q_id_key, "question_id")

    def test_invalid_file_paths(self):
        """Test error handling for invalid file paths"""
        with self.assertRaises(OSError):
            PromptDataset("nonexistent.json", self.prompt_formatter)

        with self.assertRaises(OSError):
            PromptDataset(self.qa_file, self.prompt_formatter, "nonexistent.jsonl")

    # TODO: need to correct the buffer logic first
    # def test_stsg_buffer_management(self):
    #     """Test STSG buffering functionality"""
    #     dataset = PromptDataset(self.qa_file, self.prompt_formatter, self.stsg_file)
    #     dataset.stsg_buffer_size = 2  # Small buffer for testing

    #     # Access first item should load a chunk
    #     _ = dataset[0]
    #     self.assertIn("1", dataset._stsg_buffer)
    #     self.assertIn(
    #         "5", dataset._stsg_buffer
    #     )  # Buffer size 2 should load adjacent items

    #     # Access third item should load new chunk
    #     _ = dataset[2]
    #     self.assertIn("7", dataset._stsg_buffer)


if __name__ == "__main__":
    unittest.main()
