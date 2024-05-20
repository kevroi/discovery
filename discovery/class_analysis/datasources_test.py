import unittest
import numpy as np
from discovery.class_analysis import datasources


class SeaquestDataTest(unittest.TestCase):

    def setUp(self):
        # Set up test data
        self.data = datasources.SeaquestData(
            obs_filepath="datasets/AAD/clean/SeaquestNoFrameskip-v4/episode(1).hdf5",
            label_filepath="datasets/AAD/clean/SeaquestNoFrameskip-v4/episode(1)_labels.npy",
        )

    def test_get_data(self):
        # Test the get_data() method
        obss, images, labels = self.data.get_data()

        print("obss.shape:", obss.shape)
        print("images.shape:", images.shape)
        print("labels.shape:", labels.shape)

        # Assert that the returned data is not None
        self.assertIsNotNone(obss)
        self.assertIsNotNone(images)
        self.assertIsNotNone(labels)


if __name__ == "__main__":
    unittest.main()
