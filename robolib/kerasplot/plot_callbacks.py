from keras.callbacks import Callback
import matplotlib.pyplot as plt
from threading import Thread
from threading import Event
import time


class PlotWindowThread(Thread):
    def __init__(self, loss_callback_obj):
        print("InitS")
        self.is_changed = Event()
        super().__init__()

        plt.ion()

        self.loss_callback_obj = loss_callback_obj
        fig = plt.figure()
        self.ax = fig.add_subplot(111)
        self.ydata = []

        self.ax.plot(self.ydata, 'r-')
        self.ax.set_autoscale_on(True)

        self.should_run = True
        fig.show()
        print("InitE")

    def run(self):
        print("RunS")
        super().run()
        while self.should_run:
            self.is_changed.wait()
            if len(self.ydata) <= 1:
                continue
            print("Redrawing")
            print(self.ydata)
            #self.ax.plot(self.ydata, 'r-')
            #self.ax.set_xlim(0, len(self.ydata))
            plt.plot(self.ydata)
            plt.draw()
            plt.pause(1)
            print("Redrawing done")
        print("RunE")

    def add_value(self, y):
        print("Add_value")
        self.ydata.append(y)
        self.is_changed.set()

    def last_value(self):
        print("Last_value")
        self.should_run = False
        self.is_changed.set()


class LossPlotCallback(Callback):
    def __init__(self):
        super().__init__()
        self.thread = PlotWindowThread(self)
        self.thread.start()

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        loss = (logs["loss"])
        self.add_value(loss)

    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        self.thread.last_value()

    def add_value(self, y):
        self.thread.add_value(y)
