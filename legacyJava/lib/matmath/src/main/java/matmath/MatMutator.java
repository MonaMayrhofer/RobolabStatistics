package matmath;

import org.opencv.core.Mat;

@FunctionalInterface
public interface MatMutator {
    void consume(int x, int y, Mat... m);
}