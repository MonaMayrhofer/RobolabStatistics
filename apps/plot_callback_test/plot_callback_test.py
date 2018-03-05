from robolib.kerasplot.plot_callbacks import PlotWindowThread
import time

thrd = PlotWindowThread(None)
thrd.start()

print("Starting loop")
for i in range(0, 1000):
    thrd.add_value(i)
    time.sleep(1)
    print("Current", i)
