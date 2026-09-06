"""Boundary cases and malformed trace checks, independent of MNIST matching."""
import gzip
from pathlib import Path
import tempfile
import unittest
import numpy as np
from reference import quantize,mac,output,LAYERS
from compare import read_trace

class ReferenceTests(unittest.TestCase):
    def test_signed_rounding_ties(self):
        np.testing.assert_array_equal(quantize(np.array([-1.5,-.5,.5,1.5])/1024),[-1,0,1,2])
        self.assertEqual(int(mac(0,-2,1)),0)
        self.assertEqual(int(mac(0,2,1)),1)
        self.assertEqual(int(output(-128)),0)
        self.assertEqual(int(output(128)),1)

    def test_saturation(self):
        np.testing.assert_array_equal(quantize([-100,100]),[-32768,32767])
        self.assertEqual(int(mac((1<<31)-1,4,1)),(1<<31)-1)
        self.assertEqual(int(mac(-(1<<31),-4,1)),-(1<<31))

    def test_reject_incomplete_and_duplicate(self):
        expected={k:np.zeros((1,1),dtype=np.int64) for k in LAYERS}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'trace.gz'
            for payload in ['0,conv1,0,0\n','0,conv1,0,0\n0,conv1,0,0\n']:
                p.write_bytes(gzip.compress(payload.encode()))
                with self.assertRaises(ValueError): read_trace(p,expected)

if __name__=='__main__': unittest.main()
