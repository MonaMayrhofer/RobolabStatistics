import org.opencv.core.Mat;

public interface MatMutator {
    public void consume(int x, int y, Mat... m);
}

