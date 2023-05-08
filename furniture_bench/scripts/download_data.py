"""Download datasets using rclone."""
import argparse
import subprocess


def download_folder(randomness, furniture, out_dir):
    path = f"dataset/{randomness}/{furniture}"

    print(f"Start downloading folder {path}")
    command = f"rclone copy -P furniture:{path} ./{out_dir}/{path}"  # replace 'remote' with your rclone remote name
    process = subprocess.Popen(command, shell=True)
    process.wait()
    print(f"Finished downloading folder {path}")


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
        download_folder(args.randomness, furniture, args.out_dir)


if __name__ == "__main__":
    main()
