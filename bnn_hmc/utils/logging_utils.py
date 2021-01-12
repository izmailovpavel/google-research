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
import tensorflow.compat.v2 as tf


def make_table(tabulate_dict, iteration, header_freq):
  table = tabulate.tabulate(
      [tabulate_dict.values()], tabulate_dict.keys(), tablefmt='simple',
      floatfmt='8.4f')
  table_split = table.split('\n')
  if iteration % header_freq == 0:
    table = '\n'.join([table_split[1]] + table_split)
  else:
    table = table_split[2]
  return table


def make_header(tabulate_dict):
  table = tabulate.tabulate(
      [tabulate_dict.values()], tabulate_dict.keys(), tablefmt='simple',
      floatfmt='8.4f')
  table_split = table.split('\n')
  table = '\n'.join([table_split[1]] + table_split[:2])
  return table


# def make_common_logs(
#     tf_writer, iteration, iteration_time, train_stats, test_stats
# ):
#   """Create tabulate dict and push statistics to tensorboard."""
#   tabulate_dict = OrderedDict()
#   tabulate_dict["iteration"] = iteration
#   tabulate_dict["time"] = iteration_time
#
#   with tf_writer.as_default():
#     for prefix, stats in zip(["train", "test"], [train_stats, test_stats]):
#       for key, val in stats.items():
#         if ((key not in ["likelihood", "prior"]) and
#             not (prefix == "test" and key == "log_prob")):
#           tabulate_dict["{}_{}".format(prefix, key)] = val
#         tf.summary.scalar("{}/{}".format(prefix, key), val, step=iteration)
#
#     tf.summary.scalar("telemetry/iteration_time", iteration_time,
#                       step=iteration)
#
#   return tabulate_dict
#
#
# def add_ensemle_logs(tf_writer, tabulate_dict, ensemble_stats, iteration):
#   with tf_writer.as_default():
#     for key, val in ensemble_stats.items():
#       tabulate_dict["ens_{}".format(key)] = val
#       tf.summary.scalar("ensemble/{}".format(key), val, step=iteration)
#   return tabulate_dict


def make_logging_dict(train_stats, test_stats, ensemble_stats):
  logging_dict = {}
  for prefix, stat_dict in zip(["train/", "test/", "test/ens_"],
                               [train_stats, test_stats, ensemble_stats]):
    for stat_name, stat_val in stat_dict:
      logging_dict[prefix+stat_name] = stat_val
  return logging_dict
