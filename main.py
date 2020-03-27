# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from tensorflow.python.platform import app
from autoflags import opt, autoflags

def main(unused_argv):
    Model, Model_eval = autoflags()
    if opt.trace == "":
        raise ValueError("OUT_DIR must be specified")

    if opt.num_gpus == 1:
        from train_single_gpu import train
    else:
        from train_multi_gpu import train
    train(Model, Model_eval, opt)


if __name__ == '__main__':
    app.run()
