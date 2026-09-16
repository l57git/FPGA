import unittest
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from preprocess_robust import preprocess,Config

class RobustTests(unittest.TestCase):
    def digit(self,frame=False):
        im=Image.new('L',(240,360),235);d=ImageDraw.Draw(im)
        d.ellipse((78,105,160,250),outline=25,width=8)
        if frame:
            d.rectangle((0,0,239,15),fill=5);d.rectangle((0,344,239,359),fill=5)
        return im
    def test_blank_and_frame_only_reject(self):
        for bg in [0,120,255]:
            a,d=preprocess(Image.new('L',(200,300),bg));self.assertEqual(d['status'],'no_foreground');self.assertFalse(a.any())
        im=Image.new('L',(240,360),235);d=ImageDraw.Draw(im);d.rectangle((0,0,239,12),fill=10)
        a,diag=preprocess(im);self.assertNotEqual(diag['status'],'ok');self.assertFalse(a.any())
    def test_border_does_not_change_digit(self):
        a,d=preprocess(self.digit());b,e=preprocess(self.digit(True))
        self.assertEqual(e['status'],'ok');self.assertEqual(e['frame_components_removed'],2)
        np.testing.assert_allclose(a,b,atol=1e-6)
    def test_inverted_polarity(self):
        a,_=preprocess(self.digit());b,d=preprocess(ImageOps.invert(self.digit()))
        np.testing.assert_allclose(a,b,atol=1e-6);self.assertEqual(d['polarity'],'light_on_dark')
    def test_input_unchanged_center_range(self):
        im=self.digit(True);before=im.tobytes();a,d=preprocess(im,Config(thicken=True))
        self.assertEqual(before,im.tobytes());self.assertEqual(a.shape,(28,28));self.assertEqual(a.dtype,np.float32)
        self.assertTrue(np.isfinite(a).all());self.assertGreaterEqual(a.min(),0);self.assertLessEqual(a.max(),1)
        y,x=np.indices(a.shape);self.assertAlmostEqual(float((x*a).sum()/a.sum()),13.5,delta=.6)
        self.assertAlmostEqual(float((y*a).sum()/a.sum()),13.5,delta=.6)
    def test_two_large_objects_reject(self):
        im=Image.new('L',(300,200),240);d=ImageDraw.Draw(im)
        d.ellipse((20,30,100,170),outline=10,width=8);d.ellipse((190,30,270,170),outline=10,width=8)
        a,r=preprocess(im);self.assertEqual(r['status'],'multiple_large_objects');self.assertFalse(a.any())
    def test_offcenter_digit_not_removed_as_frame(self):
        im=Image.new('L',(200,300),240);ImageDraw.Draw(im).line((15,80,15,220),fill=10,width=8)
        a,r=preprocess(im);self.assertEqual(r['status'],'ok');self.assertGreater(a.sum(),0)
    def test_small_image_error(self):
        with self.assertRaises(ValueError):preprocess(Image.new('L',(1,1)))

if __name__=='__main__':unittest.main()
