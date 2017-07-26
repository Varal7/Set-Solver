#!/usr/bin/env python

import argparse
import os
import card
import set
from queue import Queue

parser = argparse.ArgumentParser(description='Scan image for SET card')

parser.add_argument('filename', type=str, help="Filename to do something from")

from queue import Queue
from threading import Thread

class DownloadWorker(Thread):
    def __init__(self, queue, im, trained, cards):
        Thread.__init__(self)
        self.queue = queue
        self.im = im
        self.trained = trained
        self.cards = cards

    def run(self):
        while True:
            # Get the work from the queue and expand the tuple
            contour = self.queue.get()
            my_card = card.Card(self.im, contour)
            my_card.predict(self.trained)
            print(my_card)
            self.cards.append((contour, my_card))
            self.queue.task_done()

def handle_cards(filename):
    cards = []
    queue = Queue()
    im, contours = card.get_im_contours(filename)
    contours = card.filter_cards_contours(contours)
    trained = card.load_trained()
    for contour in contours:
        worker = DownloadWorker(queue, im, trained, cards)
        worker.daemon = True
        worker.start()
    for contour in contours:
        queue.put(contour)

    queue.join()
    contours, cards = zip(*cards)

    # Compute sets
    my_set = set.Set(im, contours, cards)
    # Show sets
    my_set.show_sets()
    # Write sets
    prefix = os.path.basename(args.filename).split(".")[0]
    filenames = my_set.save_sets("save", prefix)
    return filenames


if __name__ == '__main__':
    args = parser.parse_args()
    filenames = handle_cards(args.filename)
