We have added four testing datasets to the LAVIS Library. To load the data, please first run the auto-downloader to download the images:

python
Copy code
python dataset/download_scripts/download_dvqa.py
python dataset/download_scripts/download_vqdv1.py
python dataset/download_scripts/download_tallyqa.py
python dataset/download_scripts/download_tdiuc.py
These scripts will download not only the images but also the JSON files that store the question and answer pairs, placing them in the appropriate cache location.

Next, simply load the model, just like you would with any other dataset available in LAVIS:

python
Copy code
dataset = load_dataset("vqdv1_dataset")
dataset = load_dataset("tallyqa_dataset")
dataset = load_dataset("dvqa_dataset")
dataset = load_dataset("TDIUC_dataset")