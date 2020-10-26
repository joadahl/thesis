import os

print(os.getcwd())
os.makedirs(os.getcwd() + "/modelstore", exist_ok=True)