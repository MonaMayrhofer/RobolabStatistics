package matmath;

import org.opencv.core.Mat;

public interface MatMutator {
    void consume(int x, int y, Mat... m);
}