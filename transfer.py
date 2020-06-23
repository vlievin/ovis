import os
import shutil

p1 = "runs/sbm-gamma-warmup/"
p2 = "runs/sbm-gamma-warmup2/"


for f in [p for p in os.listdir(p1) if p[0] != "."]:

    if not "warmup" in f:

        exp_path = os.path.join(p1, f)

        # if exp completed
        if 'success.txt' in os.listdir(exp_path):

            if not f in os.listdir(p2):
                target_path = os.path.join(p2, f)

                print(f">>> copy {exp_path} - > {target_path}")
                shutil.copytree(exp_path, target_path)