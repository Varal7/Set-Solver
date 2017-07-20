#!/usr/bin/env python

import argparse
import os
import card
import set

parser = argparse.ArgumentParser(description='Scan image for SET card')

parser.add_argument('filename', type=str, help="Filename to do something from")

def handle_cards(filename):
    # Read cards
    im, contours = card.get_im_contours(filename)
    contours = card.filter_cards_contours(contours)
    trained = card.load_trained()
    cards = []
    for contour in contours:
        my_card = card.Card(im, contour)
        my_card.predict(trained)
        cards.append(my_card)
        print(my_card)
    # Compute sets
    my_set = set.Set(im, contours, cards)
    # Show sets
    # my_set.show_sets()
    # Write sets
    prefix = os.path.basename(args.filename).split(".")[0]
    my_set.save_sets("save", prefix)


def print_card(filename):
    im, contours = card.get_im_contours(filename)
    trained = card.load_trained()
    my_card = card.Card(im, contours[0])
    my_card.predict(trained)
    print(my_card)


if __name__ == '__main__':
    args = parser.parse_args()
    handle_cards(args.filename)
