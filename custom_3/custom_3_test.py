"""custom_3 dataset."""

import tensorflow_datasets as tfds
from . import custom_3


class Custom3Test(tfds.testing.DatasetBuilderTestCase):
  """Tests for custom_3 dataset."""
  # TODO(custom_3):
  DATASET_CLASS = custom_3.Custom3
  SPLITS = {
      'train': 3,  # Number of fake train example
      'test': 1,  # Number of fake test example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == '__main__':
  tfds.testing.test_main()
