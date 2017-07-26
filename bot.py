#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import logging
import os
import card
import functools
import set
from queue import Queue
from threading import Thread

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


INPUT_FOLDER = "raw"
SAVE_FOLDER = "save"

try:
    os.makedirs(INPUT_FOLDER)
    os.makedirs(SAVE_FOLDER)
except:
    pass

# Define a few command handlers. These usually take the two arguments bot and
# update. Error handlers also receive the raised TelegramError object in error.
def start(bot, update):
    update.message.reply_text('Hi!')
    help(bot, update)


def help(bot, update):
    update.message.reply_text('Send me a photo with at least 4 SET cards and I will try to find sets in it')


class SetWorker(Thread):
    def __init__(self, update, queue, im, trained, cards):
        Thread.__init__(self)
        self.queue = queue
        self.im = im
        self.trained = trained
        self.cards = cards
        self.update = update

    def run(self):
        while True:
            contour = self.queue.get()
            my_card = card.Card(self.im, contour)
            my_card.predict(self.trained)
            #self.update.message.reply_text(str(my_card))
            self.cards.append((contour, my_card))
            self.queue.task_done()


def handle_cards(update, filename):
    # Read cards
    queue = Queue()
    im, contours = card.get_im_contours(filename)
    contours = card.filter_cards_contours(contours)
    update.message.reply_text("I can see " + str(len(contours)) + " cards")
    update.message.reply_text("Not sure what they are, though")
    update.message.reply_text("Let me see...")
    trained = card.load_trained()
    cards = []
    for contour in contours:
        worker = SetWorker(update, queue, im, trained, cards)
        worker.daemon = True
        worker.start()

    for contour in contours:
        queue.put(contour)

    queue.join()
    contours, cards = zip(*cards)
    update.message.reply_text("Here's what I got:")
    update.message.reply_text(
            functools.reduce(lambda x,y: x + "," + y, map(str, cards))
            )

    my_set = set.Set(im, contours, cards)
    prefix = os.path.basename(filename).split(".")[0]
    filenames = my_set.save_sets(SAVE_FOLDER, prefix)
    return filenames

def show_sets(update, filenames):
    if len(filenames) == 0:
        update.message.reply_text("I've found no sets")
    elif len(filenames) == 1:
        update.message.reply_text("I've found one set")
    else:
        update.message.reply_text("And here are the " + str(len(filenames)) + " sets I've found")

    for filename in filenames:
        update.message.reply_photo(photo=open(filename, 'rb'))


def handle_photo(bot, update):
    file_id = update.message.photo[-1].file_id
    photo = bot.get_file(file_id)
    filename = os.path.join(INPUT_FOLDER, file_id[:10] + ".jpg")
    photo.download(filename)
    filenames = handle_cards(update, filename)
    show_sets(update, filenames)

def error(bot, update, error):
    logger.warn('Update "%s" caused error "%s"' % (update, error))


def main():
    # Create the EventHandler and pass it your bot's token.
    token = open("TOKEN", 'r').read().strip()
    updater = Updater(token)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))

    # on noncommand i.e message - echo the message on Telegram
    dp.add_handler(MessageHandler(Filters.photo, handle_photo))

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()
