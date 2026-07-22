import unittest

import numpy as np

from inpainting.sam2_masks_to_contract import _largest_connected_component


class Sam2MaskCleanupTest(unittest.TestCase):
    def test_largest_component_drops_isolated_speckles(self) -> None:
        mask = np.zeros((12, 16), dtype=np.bool_)
        mask[2:9, 3:10] = True
        mask[0, 0] = True
        mask[11, 15] = True

        cleaned = _largest_connected_component(mask)

        self.assertEqual(cleaned.dtype, np.bool_)
        self.assertEqual(int(cleaned.sum()), 49)
        self.assertFalse(cleaned[0, 0])
        self.assertFalse(cleaned[11, 15])


if __name__ == "__main__":
    unittest.main()
