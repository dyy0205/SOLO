import json

# with open('/versa/dyy/dataset/MSCOCO/val_9cls.json', 'r') as f:
#     data = json.load(f)
# val_imgs = data['images']
# val_anns = data['annotations']
# new_anns = []
# for ann in val_anns:
#     if ann['category_id'] == 1:
#         new_anns.append(ann)

with open('/versa/dyy/dataset/MSCOCO/train_9cls_car.json', 'r') as f:
    data = json.load(f)
imgs = data['images']
anns = data['annotations']

img_dict = {}
for img in imgs:
    img_dict[img['id']] = img['file_name']

new_anns = []
for ann in anns:
    if ann['category_id'] == 1:
        new_anns.append(ann)
print(len(new_anns))

with open('/versa/dyy/dataset/MSCOCO/pre_train.json', 'r') as f:
    data = json.load(f)
imgs += data['images']
new_anns += data['annotations']

categories = [
    {'supercategory': 'person', 'id': 1, 'name': 'person'}
]

aug_json = {"images": imgs, "annotations": new_anns, "categories": categories}
with open('/versa/dyy/dataset/MSCOCO/tc_train.json', 'w') as f:
    json.dump(aug_json, f)