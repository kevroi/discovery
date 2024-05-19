"""Moves the listed model files between machines."""

import argparse
import os

import subprocess

# from paramiko import SSHClient
# from scp import SCPClient

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
    # The / at the end is important!
    "discovery/class_analysis/results.pkl",
    "discovery/experiments/FeatAct_minigrid/model_snapshots/",
    "discovery/experiments/FeatAct_minigrid/models/",
]


def get_remote_root_dir(host, user):
    if user == "dszepesv":
        if host == "salient3":
            return "/home/dszepesv/code/discovery"
    raise ValueError(f"Unknown host/user combination: {host}/{user}")


def main():
    args = parser.parse_args()
    filesys.set_directory_in_project()
    print("Transfering files from remote machine to local machine:")
    for path in paths_to_fetch:
        full_remote_path = os.path.join(get_remote_root_dir(args.host, args.user), path)
        command = ["rsync", "-a", f"{args.host}:{full_remote_path}", path]
        print(" *", " ".join(command))
        if not args.dry_run:
            subprocess.run(command)


if __name__ == "__main__":
    main()

# with SSHClient() as ssh:
#     ssh.load_system_host_keys()
#     ssh.connect("example.com")

#     with SCPClient(ssh.get_transport()) as scp:
#         scp.put("test.txt", "test2.txt")
#         scp.get("test2.txt")
