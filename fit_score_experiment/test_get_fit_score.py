# unit-test of git_fit_score
from get_fit_score import *
import unittest

class KnownGroupings(unittest.TestCase):
	
    known_pvals = [[0.99, 0.8, 0.9, 0.7, 0.91, 0.67], \
	    [0.2, 0.9, 0.8, 0.6], [0.5, 0.6]]
    known_buffer = [0, 0, 0]
    known_c = 1
	
    def test_fit_score_known_groupings(self):
        result = get_fit_score(self.known_pvals, \
		    self.known_buffer, self.known_c)
        self.assertEqual(15.61, result)
		
if __name__ == "__main__":
    unittest.main()