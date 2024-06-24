with open("/s/ls4/users/aamore/BaseDists_ver_before_sVAE_hevyside3/configs/a3c_with_vae.ini", "r") as ff:
    lines = ff.readlines()

for line in lines:
    line=line.strip()
    if "log_dir" in line:
        res_path = line.split('=')[-1].strip()

print("../"+res_path)
