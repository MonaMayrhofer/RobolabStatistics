from keras.callbacks import Callback
import matplotlib.pyplot as plt
from threading import Thread
from threading import Event

class PlotWindowThread(Thread):
    def __init__(self, loss_callback_obj):
        self.is_changed = Event()
        super().__init__()
        self.loss_callback_obj = loss_callback_obj
        self.ax = None
        self.ydata = []
        self.should_run = True

    def run(self):
        plt.ion()

        fig = plt.figure()
        self.ax = fig.add_subplot(111)

        self.ax.plot(self.ydata, 'r-')
        self.ax.set_autoscale_on(True)

        fig.show()
        super().run()
        while self.should_run:
            self.is_changed.wait()
            self.is_changed.clear()
            if len(self.ydata) <= 1:
                continue
            plt.plot(self.ydata)
            plt.draw()
            plt.pause(0.0000001)

    def add_value(self, y):
        self.ydata.append(y)
        self.is_changed.set()

    def last_value(self):
        self.should_run = False
        self.is_changed.set()


class LossPlotCallback(Callback):
    def __init__(self):
        super().__init__()
        self.thread = PlotWindowThread(self)
        self.thread.start()

    def on_batch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        loss = (logs["loss"])
        self.add_value(loss)

    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        self.thread.last_value()

    def add_value(self, y):
        self.thread.add_value(y)
