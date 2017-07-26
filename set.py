#!/usr/bin/env python

import cv2
import numpy as np
import argparse
import os
import imutils

CARAC_NAMES = ['color', 'number', 'fill', 'shape']

def show_img(im, name=''):
    cv2.imshow(name, im)
    while True:
        k =  cv2.waitKey(0) & 0xFF
        if k == 27:
            quit()
        break
    cv2.destroyAllWindows()


class Set():
    def __init__(self, full_img, contours, cards):
        self.full_img = full_img
        self.cards = cards
        self.contours = contours
        self.sets = self.find_set()


    def check_carac(self, c_name, i, j, k):
        all_same = (self.cards[i].carac[c_name] == self.cards[j].carac[c_name] == self.cards[k].carac[c_name])
        if all_same or (self.cards[i].carac[c_name] != self.cards[j].carac[c_name] and
                        self.cards[j].carac[c_name] != self.cards[k].carac[c_name] and
                        self.cards[k].carac[c_name] != self.cards[i].carac[c_name]
                        ):
            return True
        return False

    def is_set(self, i, j, k):
        for c_name in CARAC_NAMES:
            if not self.check_carac(c_name, i, j, k):
                return False
        return True

    def find_set(self):
        numcards = len(self.cards)
        sets = []
        for i in range(numcards):
            for j in range(i):
                for k in range(j):
                    if self.is_set(i, j, k):
                        sets.append([i, j, k])
        return sets

    def draw_set(self, cur_set):
        disp = self.full_img.copy()
        cv2.drawContours(disp, self.contours, cur_set[0], (0, 255, 0), 30)
        cv2.drawContours(disp, self.contours, cur_set[1], (0, 255, 0), 30)
        cv2.drawContours(disp, self.contours, cur_set[2], (0, 255, 0), 30)
        return disp

    def show_sets(self):
        numcards = len(self.cards)
        numsets = len(self.sets)
        text = str(numcards) + " cards, " + str(numsets) + " sets"
        count = 0
        while(1):
            cur_set = self.sets[count]
            disp = self.draw_set(cur_set)
            resized = imutils.resize(disp, width=800)
            cv2.putText(resized, text, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('', resized)

            k =  cv2.waitKey(0) & 0xFF
            if k ==27:
                    break

            if k ==  ord('j') and count < numsets - 1:
                    count += 1

            if k ==  ord('k') and count > 0 :
                    count -= 1

        cv2.destroyAllWindows()

    def save_sets(self, save_folder, prefix):
        filenames = []
        try:
            os.makedirs(save_folder)
        except:
            pass

        for i in range(len(self.sets)):
            cur_set = self.sets[i]
            disp = self.draw_set(cur_set)
            resized = imutils.resize(disp, width=800)
            filename = prefix + "_" + str(i) + ".jpg"
            dest = os.path.join(save_folder, filename)
            filenames.append(dest)
            cv2.imwrite(dest, resized)
        return filenames
