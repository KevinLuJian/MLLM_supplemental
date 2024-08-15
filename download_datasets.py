import os

print("Setting up DVQA ...........")
os.system('python LAVIS/lavis/datasets/download_scripts/download_dvqa.py')
print("Setting up VQDv1 ...........")
os.system('python LAVIS/lavis/datasets/download_scripts/download_vqdv1.py')
print("Setting up TallyQA ...........")
os.system('python LAVIS/lavis/datasets/download_scripts/download_tallyqa.py')
print("Setting up TDIUC .............")
os.system('python LAVIS/lavis/datasets/download_scripts/download_tdiuc.py')
