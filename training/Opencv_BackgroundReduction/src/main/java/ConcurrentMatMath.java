import org.opencv.core.Mat;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.Function;

public class ConcurrentMatMath {

    private ExecutorService exec;
    private int maxThreads;
    public ConcurrentMatMath(int maxThreads) {
        exec = Executors.newFixedThreadPool(maxThreads);
        this.maxThreads = maxThreads;
    }

    public void mutateMat(MatMutator muator, Mat... m){
        int width = m[0].width();
        int height = m[0].height();
        for(int i = 1; i < m.length; i++){
            if(m[i].width() != width || m[i].height() != height)
                throw new IllegalArgumentException("All Mats must be equal in width and height");
        }


        //exec.execute(new MatMutatorService(0, width, 0, height, m, muator));


        int currX = 0;
        for(int i = 0; i < maxThreads; i++){
            int nextX = currX + width/maxThreads;
            exec.execute(new MatMutatorService(currX, i+width/maxThreads, 0, height, m, muator));
            currX = nextX;

        }


        exec.shutdown();
        try {
            exec.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    private class MatMutatorService implements Runnable {

        int startX;
        int endX;
        int startY;
        int endY;
        Mat[] mats;
        MatMutator mutator;

        public MatMutatorService(int startX, int endX, int startY, int endY, Mat[] mats, MatMutator mutator) {
            this.startX = startX;
            this.endX = endX;
            this.startY = startY;
            this.endY = endY;
            this.mats = mats;
            this.mutator = mutator;
        }

        @Override
        public void run() {
            for(int x = startX; x < endX; x++){
                for(int y = startY; y < endY; y++){
                    mutator.consume(x,y,mats);
                }
            }
        }
    }
}
