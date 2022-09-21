"""human_segmentation dataset."""
import tensorflow_datasets as tfds
import os
import glob
import natsort
import numpy as np
import random

# TODO(human_segmentation): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(human_segmentation): BibTeX citation
_CITATION = """
"""


class HumanSegmentation(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for human_segmentation dataset."""
  MANUAL_DOWNLOAD_INSTRUCTIONS = '/home/park/tensorflow_datasets/'
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  
  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(cornell_grasp): Specifies the tfds.core.DatasetInfo object
    
    
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'rgb': tfds.features.Image(shape=(None, None, 3)),
            'gt': tfds.features.Image(shape=(None, None, 1)),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        # supervised_keys=('input', "depth", "box"),  # Set to `None` to disable
        supervised_keys=None,
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(cornell_grasp): Downloads the data and defines the splits
    archive_path = dl_manager.manual_dir / 'human_segmentation.zip'
    extracted_path = dl_manager.extract(archive_path)

    # TODO(cornell_grasp): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(img_path=extracted_path/'rgb', mask_path=extracted_path/'gt')
    }

  def _generate_examples(self, img_path, mask_path):
    img = os.path.join(img_path, '*.jpg')
    mask = os.path.join(mask_path, '*.png')
    
    img_files = glob.glob(img)
    # img_files.sort()
    img_files = natsort.natsorted(img_files,reverse=True)
    
    mask_files = glob.glob(mask)
    # mask_files.sort()
    mask_files = natsort.natsorted(mask_files,reverse=True)

    # shuffle list same orders

    img_files = np.array(img_files)
    mask_files = np.array(mask_files)

    indices = np.arange(img_files.shape[0])
    np.random.shuffle(indices)

    img_files = list(img_files[indices])
    mask_files = list(mask_files[indices])
    
    for i in range(len(img_files)):
      yield i, {
          'rgb': img_files[i],
          'gt' : mask_files[i]
      }