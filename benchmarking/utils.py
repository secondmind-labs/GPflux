#
# Copyright (c) 2021 The GPflux Contributors.
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
#
import os
import subprocess
from typing import Mapping, Optional

from sacred import Experiment


class ExperimentName:
    """
    Uses the comments of a argument to infer the name.
    Will read the arguments from top to bottom.
    An abbreviated name can be specified between brackets.
    - Setting (Ignore) will not include the parameter in the name
    - Setting (None) will only include the value of the parameter, without its name.
    - Not specifying any name within brackets will cause the variable name to be used.
    Example:
    ```
    @ex.config
    def config():
        # Timestamp (None)
        date = datetime.datetime.now().strftime("%b%d_%H%M%S")
        # number of inducing points (M)
        num_inducing = 400
        # batch size (Nb)
        batch_size = 2048
        # Number of times the complete training dataset is iterated over (E)
        num_epochs = 200
        # Index for the output
        output = 0
        assert output in [0, 1]
        # Learning rate starting value
        lr = 1e-2
    ```
    returns:
        Sep07_170739_M-400_Nb-2048_E-200_output-0_lr-0.01
    """

    def __init__(self, experiment: Experiment, config: Mapping, prefix: Optional[str] = None):
        assert len(experiment.configurations) == 1
        self.config = config
        self.docs = experiment.configurations[0]._var_docs
        self.prefix = prefix
        self._experiment_name: Optional[str] = None

    def _build(self) -> Optional[str]:
        experiment_name = "" if self.prefix is None else self.prefix

        def _has_comment(argument_name: str) -> bool:
            return argument_name in self.docs.keys()

        def _get_abbreviated_name(argument_name: str) -> Optional[str]:
            comment = self.docs[argument_name]
            left_bracket = comment.find("(")
            right_bracket = comment.find(")")
            if left_bracket < 0 or right_bracket < 0:
                return None
            else:
                return comment[left_bracket + 1 : right_bracket]

        for k, v in self.config.items():
            prefix = "" if len(experiment_name) == 0 else "_"

            if _has_comment(k):
                abbreviated_name = _get_abbreviated_name(k)
                if abbreviated_name is None:
                    experiment_name += f"{prefix}{k}-{v}"
                elif abbreviated_name.lower() == "none":
                    experiment_name += f"{prefix}{v}"
                elif abbreviated_name.lower() == "ignore":
                    continue
                else:
                    experiment_name += f"{prefix}{abbreviated_name}-{v}"
            else:
                experiment_name += f"{prefix}{k}-{v}"

        return experiment_name

    def get(self) -> Optional[str]:
        if self._experiment_name is None:
            self._experiment_name = self._build()
        return self._experiment_name


def git_version():
    """
    Return the git revision as a string
    """
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout = subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION
