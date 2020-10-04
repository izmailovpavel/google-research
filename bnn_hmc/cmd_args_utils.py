import os
import sys


def add_common_flags(parser):
  parser.add_argument("--tpu_ip", type=str, default="10.0.0.2",
                      help="Cloud TPU internal ip "
                           "(see `gcloud compute tpus list`)")
  parser.add_argument("--seed", type=int, default=0, help="Random seed")
  parser.add_argument("--weight_decay", type=float, default=15.,
                      help="Weight decay, equivalent to setting prior std")
  parser.add_argument("--init_checkpoint", type=str, default=None,
                      help="Checkpoint to use for initialization of the chain")
  parser.add_argument("--tabulate_freq", type=int, default=40,
                      help="Frequency of tabulate table header prints")
  parser.add_argument("--dir", type=str, default=None, required=True,
                      help="Directory for checkpoints and tensorboard logs")
  parser.add_argument("--dataset_name", type=str, default="cifar10",
                      help="Name of the dataset")
  parser.add_argument("--model_name", type=str, default="lenet",
                      help="Name of the dataset")


def save_cmd(dirname):
  with open(os.path.join(dirname, "command.sh"), "w") as f:
    f.write(" ".join(sys.argv))
    f.write("\n")
