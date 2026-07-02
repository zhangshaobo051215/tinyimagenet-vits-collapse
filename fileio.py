from progress_bar import print_progress_bar
from torchvision.io import read_image

import argparse, os, pickle, torch
import pandas as pd

NUM_CLASSES = 200
IMGS_PER_CLASS = 500

"""
Suggested workflow for preparing data:
get_train_data() # construct the training set
pickle_data()    # pickle the dataset for fast reading
get_val_data()   # construct the validation set
pickle_data()    # pickle the dataset for fast reading
"""

def as_rgb(img):
    if img.shape[0] == 1:
        return img.repeat(3, 1, 1)
    return img

def get_label_mapping(data_root='.'):
    object_mapping = pd.read_csv(os.path.join(data_root, 'words.txt'), sep='\t', index_col=0, names=['label'])
    labels_str = sorted([f.name for f in os.scandir(os.path.join(data_root, 'train')) if f.is_dir()])
    labels = pd.DataFrame(labels_str, columns=['id'])
    labels['label'] = [object_mapping.loc[ids].item() for ids in labels['id']]

    return labels

def get_train_data(data_root='.'):
    images = []
    labels = []
    train_root = os.path.join(data_root, 'train')
    labels_str = sorted([f.name for f in os.scandir(train_root) if f.is_dir()])
    for class_idx, class_name in enumerate(labels_str):
        image_root = os.path.join(train_root, class_name, 'images')
        for name in sorted(os.listdir(image_root)):
            img = as_rgb(read_image(os.path.join(image_root, name)))
            images.append(img)
            labels.append(class_idx)
        print_progress_bar(class_idx + 1, len(labels_str), prefix='Progress:', suffix='Complete')

    return torch.stack(images).type(torch.ByteTensor), torch.tensor(labels, dtype=torch.long)

def get_val_data(data_root='.'):
    images = []
    labels_str = sorted([f.name for f in os.scandir(os.path.join(data_root, 'train')) if f.is_dir()])
    labels = []
    val_root = os.path.join(data_root, 'val')
    val_annotations = pd.read_csv(os.path.join(val_root, 'val_annotations.txt'), sep='\t', names=['filename', 'label_str', 'x_min', 'y_min', 'x_max', 'y_max'])
    val_images_root = os.path.join(val_root, 'images')
    val_files = sorted(os.listdir(val_images_root))
    
    for i, name in enumerate(val_files, 1):
        img = as_rgb(read_image(os.path.join(val_images_root, name)))
        images.append(img)
        class_name = val_annotations.loc[val_annotations['filename'] == name]['label_str'].item()
        labels.append(labels_str.index(class_name))
        print_progress_bar(i, len(val_files), prefix='Progress:', suffix='Complete')

    return torch.stack(images).type(torch.ByteTensor), torch.tensor(labels, dtype=torch.long)

def pickle_data(data, label, filename):
    outfile = open(filename, 'wb')
    pickle.dump((data, label), outfile)
    outfile.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='.', help='Tiny-ImageNet root containing train, val, words.txt')
    parser.add_argument('--output-dir', default=None, help='Where to write train_dataset.pkl and val_dataset.pkl')
    args = parser.parse_args()
    output_dir = args.output_dir or args.data_root
    os.makedirs(output_dir, exist_ok=True)

    data, labels = get_train_data(args.data_root)
    print(data.shape, labels.shape)
    pickle_data(data, labels, os.path.join(output_dir, 'train_dataset.pkl'))
    data, labels = get_val_data(args.data_root)
    print(data.shape, labels.shape)
    pickle_data(data, labels, os.path.join(output_dir, 'val_dataset.pkl'))
