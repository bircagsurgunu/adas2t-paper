from datasets import load_dataset
from itertools import islice
# Note this will take very long time to download and preprocess
# you can try small subset for testing purpose
ds =load_dataset("facebook/voxpopuli", "en_accented",split = "test")


print(ds[0])
