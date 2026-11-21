import copy,unittest,sys,pathlib
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from compare import gen,condition,N,BANDS,ZipperTracks,TinyDDPM,Diffuser

class OcclusionTest(unittest.TestCase):
    def test_source_mask_and_same_input(self):
        s=gen(50001)
        self.assertEqual((N,2,24,24),s['obs'].shape)
        self.assertTrue(np.all(s['obs'][:,:,:,BANDS[0]:BANDS[1]]==0))
        self.assertTrue(np.any(s['clean'][:,:,:,BANDS[0]:BANDS[1]]>.4))
        self.assertFalse(np.shares_memory(s['obs'],s['clean']))
    def test_reproducible_test_seed(self):
        a,b=gen(50001),gen(50001)
        np.testing.assert_array_equal(a['obs'],b['obs']);np.testing.assert_array_equal(a['truth'],b['truth'])
    def test_future_blind_condition(self):
        s=gen(50001);x=condition(s,6).copy()
        s['obs'][7:]=999
        np.testing.assert_array_equal(x,condition(s,6))
    def test_reversible_seam_does_not_touch_originals(self):
        s=gen(50000);z=ZipperTracks()
        for frame in s['obs'][:12]:z.step(frame)
        before=z.original_tape[4].copy();self.assertTrue(z.link_masks[4][0])
        z.unzip(4,0);self.assertFalse(z.link_masks[4][0])
        z.rezip(4,0);self.assertTrue(z.link_masks[4][0])
        np.testing.assert_array_equal(before,z.original_tape[4])
    def test_models_are_not_equivalent(self):
        net=TinyDDPM();self.assertGreater(sum(p.numel() for p in net.parameters()),1000)
        s=gen(50002)
        x=condition(s,8)
        self.assertEqual(x.shape,(7,24,24))
    def test_causal_track_past(self):
        s=gen(50001);a=ZipperTracks();b=ZipperTracks()
        for frame in s['obs'][:7]:a.step(frame);b.step(frame)
        previous=copy.deepcopy(a.archive)
        s['obs'][7:]=0
        for frame in s['obs'][7:]:b.step(frame)
        self.assertEqual(previous,b.archive[:7]);self.assertEqual(len(a.original_tape),7)

if __name__=='__main__':unittest.main()
