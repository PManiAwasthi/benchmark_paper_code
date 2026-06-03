import pickle

path = "datasets/MOSEI/Processed/aligned_50.pkl"

with open(path, "rb") as f:
    data = pickle.load(f)

print(type(data))
print(data.keys())

# dict_keys(['train','valid','test']) expected output
# check confirmed


#step 2 to instpect the content
for split in ["train", "valid", "test"]:
    print(f"Split: {split}")
    print(data[split].keys())

# expected output raw_text, audio, vision, text, text_bert etc
# got 'raw_text', 'audio', 'vision', 'id', 'text', 'text_bert', 'annotations', 'classification_labels', 'regression_labels'
#step 3 we check the data
print(len(data["train"]["audio"]))
print(len(data["train"]["text"]))
print(len(data["train"]["vision"]))
#got output 1284 for all three which is expected as they should be the same length

# step 4 we then check shape of a specific sample
sample_audio = data["train"]["audio"][0]
print(sample_audio.shape)

sample_text = data["train"]["text"][0]
print(sample_text.shape)

sample_vision = data["train"]["vision"][0]
print(sample_vision.shape)

label = data["train"]["regression_labels"][0]
print(label)

#got output (50, 5) (50, 768) (50, 20) -0.5


# step 5 we check if the splits of train, text and valid are similar in shaped or not
print(data["train"]["text"].shape)
print(data["valid"]["text"].shape)
print(data["test"]["text"].shape)

print(data["train"]["audio"].shape)
print(data["valid"]["audio"].shape)
print(data["test"]["audio"].shape)

print(data["train"]["vision"].shape)
print(data["valid"]["vision"].shape)
print(data["test"]["vision"].shape)


print(data["train"]["id"][0])
print(data["train"]["id"][1])
print(data["train"]["id"][2])

print(data["train"]["raw_text"][0])
# got output
# (1284, 50, 768)
# (229, 50, 768)
# (686, 50, 768)
# (1284, 50, 5)
# (229, 50, 5)
# (686, 50, 5)
# (1284, 50, 20)
# (229, 50, 20)
# (686, 50, 20)
# 03bSnISJMiM$_$11
# 03bSnISJMiM$_$10
# 03bSnISJMiM$_$13
# A LOT OF SAD PARTS