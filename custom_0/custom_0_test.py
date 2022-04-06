"""custom_0 dataset."""

import tensorflow_datasets as tfds
from . import custom_0


class Custom0Test(tfds.testing.DatasetBuilderTestCase):
  """Tests for custom_0 dataset."""
  # TODO(custom_0):
  DATASET_CLASS = custom_0.Custom0
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
