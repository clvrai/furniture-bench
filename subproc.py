import subprocess

def main():

    # furnitures = ['lamp', 'chair', 'desk', 'square_table', 'round_table', 'cabinet', 'drawer', 'one_leg', 'stool']
    furnitures = ['cabinet']

    cmds = []
    for furniture in furnitures:

        cmds.append(f'python furniture_bench/scripts/run_sim_env.py --furniture {furniture} --no-action --from-skill 0 --num-envs 1 --save-camera-input --init-assembled')

    for cmd in cmds:
        subprocess.call(cmd, shell=True)

if __name__ == '__main__':
    main()
