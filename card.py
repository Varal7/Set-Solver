#!/usr/bin/env python

import cv2
import numpy as np
import argparse
import os
import imutils

parser = argparse.ArgumentParser(description='Scan image for SET card')

parser.add_argument('filename', type=str, help="Filename to do something from")


TRAIN_FOLDER = "train"
CANVAS_SIZE = 500
EXTENSIONS = ['.png', '.jpg']
number_to_int = {'one': 1, 'two': 2, 'three': 3}
boundaries = {
        'green': [([40, 50, 50], [80, 255, 255])],
        'red': [([0, 50, 50], [10, 255, 255]), ([175, 50, 50], [255, 255, 255])],
        'purple': [([120, 50, 50], [165, 255, 255])],
}
carac_names = ['color', 'number', 'fill', 'shape']

FILL_THRESHOLD_OUTLINED = 10000
PREPROCESS_FILL_THRESHOLD = 150


def imgdiff(img1, img2):
    """Takes as input two preprocessed images and returns diff"""
    kernel = np.ones((5,5),np.uint8)
    diff = cv2.absdiff(img1, img2)
    diff = cv2.GaussianBlur(diff, (5, 5), 5)
    flag, diff = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY)
    return np.sum(diff)


def get_carac_from_filename(filename):
    base = filename.split(".")[0]
    parts = base.split("_")
    return {
            'shape': parts[0],
            'color': parts[1],
            'fill': parts[2],
            'number': parts[3],
            }


class Card():
    def __init__(self, full_img, contour):
        self.card = self.get_card(full_img, contour)
        self.prep_shape, self.prep_fill = self.get_preprocessed()
        self.closest = None
        self.carac = {
            'shape': None,
            'color': None,
            'fill': None,
            'number': None,
            }


    def get_card(self, img, contour):
        disp = img.copy()
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
        canvas = np.array(
                [[0, 0], [CANVAS_SIZE, 0], [CANVAS_SIZE, CANVAS_SIZE], [0, CANVAS_SIZE]],
                np.float32)
        approx = approx.astype(np.float32)
        transform = cv2.getPerspectiveTransform(approx, canvas)
        card = cv2.warpPerspective(disp, transform,(CANVAS_SIZE, CANVAS_SIZE))
        return card

    def get_preprocessed(self):
        gray = cv2.cvtColor(self.card, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 2 )

        #Preprocess for shape
        prep_shape = cv2.adaptiveThreshold(blur, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)
        kernel = np.ones((5,5),np.uint8)
        prep_shape = cv2.erode(prep_shape, kernel, iterations = 1)
        prep_shape = cv2.GaussianBlur(prep_shape, (5, 5), 5)

        #Preprocess for fill
        _, prep_fill = cv2.threshold(blur, PREPROCESS_FILL_THRESHOLD, 255, cv2.THRESH_BINARY)

        return prep_shape, prep_fill


    def imgdiff(self, other_preprocessed):
        return imgdiff(self.prep_shape, other_preprocessed)

    def predict_from_closest(self, gold):
        order = sorted(gold, key=lambda x:self.imgdiff(x["image"]))
        self.closest = order[0]
        name = self.closest["name"]
        self.carac = get_carac_from_filename(name)

    def predict_color(self):
        color = "unkown"
        image = self.card
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        maxi = -1
        for cur_color, bounds in boundaries.items():
            mask = np.zeros(image.shape[:2], dtype = "uint8")
            for (lower, upper) in bounds:
                lower = np.array(lower, dtype = "uint8")
                upper = np.array(upper, dtype = "uint8")
                cur_mask = cv2.inRange(hsv, lower, upper)
                mask += cur_mask
            output = cv2.bitwise_and(image, image, mask = mask)
            #output_hsv = cv2.bitwise_and(hsv, hsv, mask = mask)
            count = np.count_nonzero(output)

            if count > maxi:
                maxi = count
                color = cur_color

        self.carac['color'] = color

    def predict_fill(self, number):
        total = CANVAS_SIZE * CANVAS_SIZE
        img = self.prep_fill
        count = total - np.count_nonzero(img)
        if count > FILL_THRESHOLD_OUTLINED * number:
            self.carac['fill'] = "solid"
        else:
            self.carac['fill'] = "outlined"

    def predict(self, trained):
        self.predict_from_closest(trained)
        number = number_to_int[self.carac['number']]
        if self.carac['fill'] in ['outlined', 'solid']:
            self.predict_fill(number)

        self.predict_color()

    def __str__(self):
        string = self.carac['shape']
        string += '_' + self.carac['color']
        string += '_' + self.carac['fill']
        string += '_' + self.carac['number']
        return string


def get_im_contours(filename):
    im = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (1, 1), 1000)
    flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea,reverse=True)
    return im, contours

def filter_cards_contours(contours):
    area_mean = sum(cv2.contourArea(contours[i]) for i in range(1, 4)) / 3.
    contours = [contour for contour in contours if 2 * area_mean > cv2.contourArea(contour) > area_mean / 2]
    numcards = len(contours)
    print("Num cards: " + str(numcards))
    return contours

def load_trained():
    trained = []
    for filename in os.listdir(TRAIN_FOLDER):
        extension = os.path.splitext(filename)[1]
        if extension.lower() not in EXTENSIONS:
            continue
        train_path = os.path.join(TRAIN_FOLDER, filename)
        im = cv2.imread(train_path, cv2.IMREAD_GRAYSCALE)
        trained.append({'name': filename, 'image': im})
    return trained

if __name__ == '__main__':
    args = parser.parse_args()

    # Predict card
    im, contours = get_im_contours(args.filename)
    card = Card(im, contours[0])
    card.predict()
    print(card)
