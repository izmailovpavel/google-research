# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Utility functions printing tabulated outputs."""

import tabulate


def make_table(tabulate_dict, iteration, header_freq):
  table = tabulate.tabulate(
      [tabulate_dict.values()], tabulate_dict.keys(), tablefmt='simple',
      floatfmt='8.7f')
  if iteration % header_freq == 0:
    table = table.split('\n')
    table = '\n'.join([table[1]] + table)
  else:
    table = table.split('\n')[2]
  return table