# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Add HuggingFace Datasets.

Script to convert code for dataset in HuggingFace to tensorflow datasets

"""

import argparse
import os
import pathlib
import re

from absl import app
from absl import flags
from absl.flags import argparse_flags
import tensorflow.compat.v2 as tf
from tensorflow_datasets.core.utils import py_utils
from tensorflow_datasets.scripts.cli import new

# In TF 2.0, eager execution is enabled by default
tf.compat.v1.disable_eager_execution()

flags.DEFINE_string("tfds_dir", py_utils.tfds_dir(),
                    "Path to tensorflow_datasets directory")
FLAGS = flags.FLAGS


TO_CONVERT = [
    # (pattern, replacement)
    # Order is important here for some replacements
    (r"from\s__future.*", r""),
    (r"import\slogging", r"from absl import logging\n"),
    (r"import\snlp",
     r"import tensorflow as tf\nimport tensorflow_datasets.public_api as tfds\n"
    ),
    (r"with\sopen", r"with tf.io.gfile.GFile"),
    (r"nlp\.Value\(\"string\"\)", r"tfds.features.Text()"),
    (r"nlp\.Value\(\"string\"\),", r"tfds.features.Text("),
    (r"nlp\.Value\(\"([\w\d]+)\"\)", r"tf.\1"),
    (r"nlp\.features", "tfds.features"),
    (r"features\s*=\s*nlp\.Features\(",
     r"features=tfds.features.FeaturesDict("),
    (r"dict\(", r"tfds.features.FeaturesDict("),
    (r"nlp.SplitGenerator", r"tfds.core.SplitGenerator"),
    (r"self\.config\.data_dir", r"dl_manager.manual_dir"),
    (r"self\.config", r"self.builder_config"),
    (r"nlp\.Split", r"tfds.Split"),
    (r"nlp", r"tfds.core"),
]


def _parse_flags(_) -> argparse.Namespace:
  """Command line flags."""
  parser = argparse_flags.ArgumentParser(
      prog="convert_dataset",
      description="Tool to add hugging face datasets",
  )
  parser.add_argument(
      "--dataset_name",
      help="Path of hugging face.",
  )
  parser.add_argument(
      "--library",
      help="Path of tensorflow",
  )
  return parser.parse_args()


def main(args: argparse.Namespace):

  create_dataset_files(
      dataset_name=args.dataset_name, library=args.library)


def create_dataset_files(dataset_name: str = None,
                         library: str = None) -> None:
  """Create template files."""
  root_dir = pathlib.Path(os.path.join(FLAGS.tfds_dir, library, dataset_name))
  root_dir.mkdir(parents=True)

  #  Create dataset timeplate files from new.py
  new.create_dataset_files(dataset_name, True, root_dir)

  nlp_file_path = os.path.join(FLAGS.tfds_dir, "scripts", "cleanup", "nlp.py")
  converted_file_path = root_dir / "{}.py".format(dataset_name)

  #  Convert nlp dataset --> tfds
  convert_dataset_file(nlp_file_path, converted_file_path)


def convert_dataset_file(nlp_file_path, converted_file_path):
  """Convert nlp dataset."""
  compiled_patterns = [
      (re.compile(pattern), replacement) for pattern, replacement in TO_CONVERT
  ]

  with tf.io.gfile.GFile(nlp_file_path, "r") as f:
    lines = f.readlines()

    out_lines = []
    for line in lines:
      out_line = line

      if "return nlp.DatasetInfo(" in out_line:
        out_lines.append("    return tfds.core.DatasetInfo(\n")
        out_lines.append("        builder=self,\n")
        out_line = ""
      else:
        for pattern, replacement in compiled_patterns:
          out_line = pattern.sub(replacement, out_line)

      assert ("nlp" not in out_line), f"Error converting {out_line.strip()}"
      out_lines.append(out_line)

      with open(converted_file_path, "w") as f:
        f.writelines(out_lines)


if __name__ == "__main__":
  app.run(main, flags_parser=_parse_flags)
