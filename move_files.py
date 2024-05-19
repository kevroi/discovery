"""Moves the listed model files between machines."""

import argparse
import os

from paramiko import SSHClient
from scp import SCPClient

from discovery.utils import filesys


parser = argparse.ArgumentParser()
parser.add_argument(
    "--host",
    type=str,
    required=True,
    help="The host from which to fetch the files.",
)
parser.add_argument(
    "--user",
    type=str,
    required=True,
    help="dszepesv or roice.",
)
parser.add_argument(
    "--dry_run",
    action="store_true",  # set to false if we do not pass this argument
    help="Just list what would be happening.",
)


paths_to_fetch = [
    "discovery/class_analysis/results.pkl",
    "discovery/experiments/FeatAct_minigrid/model_snapshots",
    "discovery/experiments/FeatAct_minigrid/models",
]


def get_root_dir(host, user):
    if user == "dszepesv":
        if host == "salient":
            return "/home/dszepesv/code/discovery"
    raise ValueError(f"Unknown host/user combination: {host}/{user}")


def main():
    args = parser.parse_args()
    filesys.set_directory_in_project()

    with SSHClient() as ssh:
        ssh.load_system_host_keys()
        ssh.connect(args.host)

        with SCPClient(ssh.get_transport()) as scp:
            get_all_files(args, scp)


def get_all_files(args, scp):
    for path in paths_to_fetch:
        full_remote_path = os.path.join(get_root_dir(args.host, args.user), path)
        print(f"Copy recursively {full_remote_path} --> {path}")
        if not args.dry_run:
            scp.get(
                full_remote_path, local_path=path, recursive=True, preserve_times=True
            )


if __name__ == "__main__":
    main()

# with SSHClient() as ssh:
#     ssh.load_system_host_keys()
#     ssh.connect("example.com")

#     with SCPClient(ssh.get_transport()) as scp:
#         scp.put("test.txt", "test2.txt")
#         scp.get("test2.txt")
