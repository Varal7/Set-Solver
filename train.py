import argparse
import os
import card
import cv2

parser = argparse.ArgumentParser(description='Scan image for SET card')

parser.add_argument('--photos-folder', required=True, help="Folder with images files")
parser.add_argument( '--train-folder', default="train", help="Train folder with .txt files")

EXTENSIONS = ['.png', '.jpg']

args = parser.parse_args()

def train(photos_folder, train_folder):
    for filename in os.listdir(photos_folder):
        extension = os.path.splitext(filename)[1]
        if extension.lower() not in EXTENSIONS:
            continue
        print(filename)
        photo_path = os.path.join(photos_folder, filename)
        dest_path = os.path.join(train_folder, filename)
        im, contours = card.get_im_contours(photo_path)
        my_card = card.Card(im, contours[0])
        prep_shape = my_card.prep_shape
        cv2.imwrite(dest_path, prep_shape)
    print("Done")


if __name__ == '__main__':
    train(args.photos_folder, args.train_folder)
