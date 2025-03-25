import re

def sum_html_comment_lengths(text: str) -> int:
    """
    Find all content enclosed in HTML comments (<!-- -->) and 
    return the sum of their lengths.
    
    Args:
        text: Input string to search for HTML comments
        
    Returns:
        The sum of lengths of all content within HTML comments
    """
    # Find all matches of <!-- ... --> pattern, including multiline comments
    pattern = r'<!--(.*?)-->'
    matches = re.findall(pattern, text, re.DOTALL)
    
    # Calculate the sum of lengths
    total_length = sum(len(match) for match in matches)
    
    return total_length

import unittest
from x_r1.dataset.utils.text_analysis import sum_html_comment_lengths

class TestSumHTMLCommentLengths(unittest.TestCase):
    
    def test_no_comments(self):
        """Test with a string that has no HTML comments."""
        text = "This is a string without any HTML comments"
        result = sum_html_comment_lengths(text)
        self.assertEqual(result, 0)
    
    def test_single_comment(self):
        """Test with a string that has a single HTML comment."""
        text = "Some text <!-- comment --> and more text"
        result = sum_html_comment_lengths(text)
        self.assertEqual(result, 7)  # "comment" is 7 characters
    
    def test_multiple_comments(self):
        """Test with a string that has multiple HTML comments."""
        text = "Start <!-- first --> middle <!-- second --> end"
        result = sum_html_comment_lengths(text)
        self.assertEqual(result, 11)  # "first" (5) + "second" (6) = 11
        
    def test_multiline_comment(self):
        """Test with a multi-line HTML comment."""
        text = """Before
        <!-- multi
        line
        comment --> After"""
        result = sum_html_comment_lengths(text)
        # Count actual characters including newlines and spaces
        expected = len("multi\n        line\n        comment")
        self.assertEqual(result, expected)
    
    def test_empty_comments(self):
        """Test with empty HTML comments."""
        text = "Some text <!----> with an empty comment"
        result = sum_html_comment_lengths(text)
        self.assertEqual(result, 0)
    
    def test_html_inside_comment(self):
        """Test with HTML-like content inside comments."""
        text = "Testing <!-- <div>Some HTML</div> --> here"
        result = sum_html_comment_lengths(text)
        self.assertEqual(result, 23)  # Length of "<div>Some HTML</div> "

if __name__ == "__main__":
    unittest.main()


