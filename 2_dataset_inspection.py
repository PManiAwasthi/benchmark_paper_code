import pickle

path = "datasets/MOSI/Processed/aligned_50.pkl"

with open(path, "rb") as f:
    data = pickle.load(f)

print(type(data))
print(data.keys())

# dict_keys(['train','valid','test']) expected output

