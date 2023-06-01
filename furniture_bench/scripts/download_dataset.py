"""Download datasets using rclone or gdown."""
import os
import argparse
import subprocess


def download_folder_rclone(randomness, furniture, out_dir):
    """Use rclone to download a folder from the Google Drive."""
    path = f"dataset/{randomness}_compressed/{furniture}.tar.gz"

    print(f"Start downloading folder {path}")
    command = f"rclone copy -P furniture:{path} ./{out_dir}/{randomness}"
    process = subprocess.Popen(command, shell=True)
    process.wait()
    print(f"Finished downloading folder {path}")


def download_file_gdown(randomness, furniture, out_dir):
    "use gdown to download a file from the Google Drive."
    import gdown

    download_links = {
        "high": {
            "cabinet": "1RHbo27phzXVJDjMXPI91QKPHWByCSQU4",
            "chair": "1D8j1s4v9NL02V03PwEU6moOjn9v5qzFn",
            "desk": "1xOhzI96-BORgjyqF7rBdYRPc_-RxYr_Z",
            "drawer": "1QNqA48y9fFE4251xMmCMaUOdmNcnaZ8T",
            "lamp": "1Ia0SHIACIoqwzVjhc_dgzEMsiKffIO6T",
            "one_leg": "19iLUPDAvrRzevVggfD09nK2ayZ3GrVfC",
            "round_table": "1qS2lIiPdqq8pTJsPN2txU1lzQJ4seWCr",
            "square_table": "1Wq1MVCUSXxi6wJk7CQW-3LMGdUGJdiAR",
            "stool": "1QeEb4Ajz-qN820Y_DVuxU7UdvrZBv7hJ",
        },
        "med": {
            "cabinet": "1LRexjymeP0szZucTEt40ZL-2VoLYXKKo",
            "chair": "1wXGloFr4aVJ3ChYz4qKD_zRheuivM9rc",
            "desk": "1edLqFAxKRAPcnNgDkRBmw9AilN8zZSqs",
            "drawer": "1nFdVpUERi90zNNthfOR2sdYCjrW7Rg_c",
            "lamp": "1awqLazZlNOqDhnOuElttOwDOol9oWY0C",
            "one_leg": "1zRqpz3WLztpOo7ULYC6Ik3rWyYtoo9ch",
            "round_table": "1gJ_HmhpgE4nJNBMmEKHHx7mKjYUQadRA",
            "square_table": "1T4QLiCaJQjzsLANUR8jPssGsVZFChMog",
            "stool": "1IstEhReeRri2s2y7vJrcv1oQ3wUm4kqT",
        },
        "low": {
            "cabinet": "1zjMLlRlXVZDGri1QUINV540DTG7-jAa-",
            "chair": "1swulRnjB7rU1u-TuG6-WOrci9o8WZaEI",
            "desk": "1aJEqENTUCvnHhoAlwd9rks38YzkgeecL",
            "drawer": "121seXYws04z3UowpUdT-7uxg-y0l4Qb2",
            "lamp": "1kD9Fxj49Df4mgZPVkBa_b_L3dqzEQROF",
            "one_leg": "1E121w1Q9-SzFN3Bf6wC_NF-7kDA57RZf",
            "round_table": "1SjSg2tzQZ4fsN6z_xLT1vEOuTTzrQCun",
            "square_table": "1ogI5VkFcGeJsFje9_0AS_fFSwhJX6zQR",
            "stool": "1Z1ewa62pkWehC4biodoDdPfWDJdCNPtb",
        },
    }

    if not os.path.exists(f"{out_dir}/{randomness}"):
        os.makedirs(f"{out_dir}/{randomness}")

    print(f"Start downloading file {randomness}/{furniture}")

    gdown.download(
        id=download_links[randomness][furniture],
        output=f"{out_dir}/{randomness}/{furniture}.tar.gz",
        quiet=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--randomness", help="Randomness of initial state", required=True
    )
    parser.add_argument(
        "--furniture",
        help="Name of the furniture. --all to download all furniture datasets.",
        required=True,
    )
    parser.add_argument("--out_dir", help="Path to output directory", required=True)
    parser.add_argument(
        "--use-rclone", action="store_true", help="Use rclone to download data"
    )
    parser.add_argument(
        "--untar", action="store_true", help="Untar the downloaded file"
    )
    args = parser.parse_args()

    if args.furniture == "all":
        download_list = [
            "lamp",
            "square_table",
            "desk",
            "round_table",
            "stool",
            "chair",
            "drawer",
            "cabinet",
            "one_leg",
        ]
    else:
        download_list = [args.furniture]

    for furniture in download_list:
        if args.use_rclone:
            download_folder_rclone(args.randomness, furniture, args.out_dir)
        else:
            download_file_gdown(args.randomness, furniture, args.out_dir)

        if args.untar:
            print(f"Untarring {furniture}")
            command = f"tar -xvzf {args.out_dir}/{args.randomness}/{furniture}.tar.gz -C {args.out_dir}/{args.randomness}"
            process = subprocess.Popen(command, shell=True)
            process.wait()
            print(f"Finished untarring {furniture}")


if __name__ == "__main__":
    main()
