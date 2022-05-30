import os
import tqdm
import cv2

DATA_FOLDER = './data/'
IMAGES_FOLDER = 'raw_images/'
CROPED_FOLDER = 'images/'


def crop_image(data_path,):
    image_path = os.path.join(data_path, IMAGES_FOLDER)
    cropped_path = os.path.join(data_path, CROPED_FOLDER)
    box_info = os.path.join(data_path, 'images_box.txt')

    if not os.path.exists(cropped_path):
        os.makedirs(cropped_path)

    with open(box_info, 'r') as f:
        lines = f.read().splitlines()
        
        for line in tqdm.tqdm(lines):
            line = line.split(' ')
            image_name = line[0] + '.jpg'
            if not os.path.exists(os.path.join(image_path, image_name)):
                print('Image not found: {}'.format(image_name))
                continue
            image = cv2.imread(os.path.join(image_path, image_name))
            image = image[int(line[2]):int(line[4]), int(line[1]):int(line[3])]  # xmin, ymin, xmax and ymax
            try:
                cv2.imwrite(os.path.join(cropped_path, image_name), image)
            except AssertionError:
                print('Image not found: {}'.format(image_name))


def extract_labels(data_path):
    label_path = os.path.join(data_path, 'image_variant.txt')
    labels_path = os.path.join(data_path, 'labels')

    if not os.path.exists(labels_path):
        os.makedirs(labels_path)

    if not os.path.exists(label_path):
        val_label = os.path.join(data_path, 'images_variant_val.txt')
        train_label = os.path.join(data_path, 'images_variant_train.txt')
        test_label = os.path.join(data_path, 'images_variant_test.txt')

        file_names = [val_label, train_label, test_label]
        with open(label_path, 'w') as outfile:
            for fname in file_names:
                with open(fname, 'r') as infile:
                    outfile.write(infile.read())

                outfile.write('\n')

    labels = []
    with open(label_path, 'r') as f:
        lines = f.read().splitlines()

        for line in tqdm.tqdm(lines):
            line = line.split(' ')
            image_name = line[0]
            label = ' '.join(line[1:])
            
            with open(os.path.join(labels_path, image_name + '.txt'), 'w') as f:
                f.write(label)

if __name__ == '__main__':
    # crop_image(DATA_FOLDER,)
    extract_labels(DATA_FOLDER)
    pass