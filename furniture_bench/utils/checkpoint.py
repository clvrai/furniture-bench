import os
import subprocess


def download_ckpt_if_not_exists(ckpt_dir, run_name, seed=None):
    run_name_to_ckpt = {
        f"one_leg_full_iql_r3m_low_sim_1000.{seed}": "1Jo2hG26tQgj9HmNtigg8Za2zG64Deb6R",
        f"one_leg_full_iql_r3m_low_1000.{seed}": "1sfG_MaZdhq_-utBdqin3icGpFHyttxbS",
        f"one_leg_full_iql_r3m_med_1000.{seed}": "1w3mCdRdARmQWJiR-i_e8lwmzY2aQ7zjG",
        f"one_leg_full_iql_r3m_mixed_2000.{seed}": "1FiA5pCBM-fF2MPj5fyKxMrHN6Vx0iSah",
        f"desk_full_iql_r3m_low_100.{seed}": "1sxHl7FyQaUKm9I5WCyWfvY2LJviOdEUD",
        f"drawer_full_iql_r3m_low_250.{seed}": "1K_rHm25aDb8LyFb95yUK4iVUmwB_NB_D",
        f"lamp_full_iql_r3m_low_150.{seed}": "1GM7ikLouLhY3UHPhTouGYcxy_nqxFPb3",
        f"round_table_full_iql_r3m_low_100.{seed}": "1e5pZxW0L29MgmuC_KYF3Suofb5oGUDV3",
        f"square_table_full_iql_r3m_low_150.{seed}": "1zY1eswvasJJ96cGoa7j2eo86VC2Ow6j6",
        f"stool_full_iql_r3m_low_100.{seed}": "12Yh79T1QHNrVClhNt3OIrdVL3ruKtLyR",
        f"cabinet_full_iql_r3m_low_150.{seed}": "18Yg2PTK8iymlmeb3vg-PAFI6z8wxw1RW",
        "one_leg_full_bc_resnet18_low_sim_1000": "1ROy2bEetZ7-sv1CPBZXrQyHkj9mMMZnK",
    }

    if seed is None:
        ckpt_dir = os.path.join(ckpt_dir, run_name)
    else:
        ckpt_dir = os.path.join(ckpt_dir, f"{run_name}.{seed}")
        run_name = f"{run_name}.{seed}"

    if not os.path.exists(ckpt_dir):
        if run_name in run_name_to_ckpt:
            print("Downloading checkpoints...")

            os.makedirs(ckpt_dir, exist_ok=True)
            file_id = run_name_to_ckpt[run_name]
            wget_cmd = f'wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id={file_id}" -O- | sed -rn "s/.*confirm=([0-9A-Za-z_]+).*/\1\n/p")&id={file_id}" -O {ckpt_dir}.tar.gz && rm -rf ~/cookies.txt'

            try:
                subprocess.run(wget_cmd, check=True, shell=True)
            except subprocess.CalledProcessError as e:
                raise Exception(f"Failed to download the tar file. Error: {e}")

            tar_cmd = f"tar -xvzf {ckpt_dir}.tar.gz -C {ckpt_dir} --strip-component=1"
            rm_cmd = f"rm {ckpt_dir}.tar.gz"

            try:
                subprocess.run(tar_cmd, check=True, shell=True)
            except subprocess.CalledProcessError as e:
                raise Exception(f"Failed to extract the tar file. Error: {e}")

            subprocess.run(rm_cmd, check=True, shell=True)
