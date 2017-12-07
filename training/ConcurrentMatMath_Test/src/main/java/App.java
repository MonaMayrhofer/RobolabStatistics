/*
Erik Mayrhofer - Proof Of Concept of Background-Reduction
 */
import matmath.ConcurrentMatMath;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import robogui.RoboGui;

import javax.swing.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.lang.reflect.Field;

public class App {

    private static boolean running = true;

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) throws InterruptedException{
        int thresh = 3; //Play around with this value


        VideoCapture capt = new VideoCapture();
        capt.open(0);

        RoboGui.getInstance().addWindowListener("Image", new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) { running = false; }
        });

        Mat currImg = new Mat();
        Mat rawestImg = new Mat();
        Mat rawImg = new Mat();
        Mat oldImg = new Mat();
        Mat oldestImg;

        capt.read(oldImg);

        //oldImg = new Mat(1000,1000, CvType.CV_8UC3);
        RoboGui.getInstance().setFrameSize("Image", oldImg.width(), oldImg.height());
        Mat actImg = new Mat(oldImg.height(), oldImg.width(), oldImg.type(), Scalar.all(0));
        Imgproc.cvtColor(oldImg, oldImg, Imgproc.COLOR_RGB2GRAY);

        long min = 100;
        long max = 0;
        double avg = 0.0;

        while(running){
            actImg.setTo(Scalar.all(255));
            capt.read(rawestImg);

            oldestImg = oldImg.clone();

            Imgproc.cvtColor(rawestImg, rawImg, Imgproc.COLOR_RGB2GRAY);
            Core.multiply(oldImg, Scalar.all(0.9), oldImg);
            Core.multiply(rawImg, Scalar.all(0.1), currImg);
            Core.add(currImg, oldImg, oldImg);

            long start = System.currentTimeMillis();
            ConcurrentMatMath m = new ConcurrentMatMath(8); //Anzahl der Threads hier einfügen


            m.mutateMat((x, y, ci) -> {
                double[] curr = ci[0].get(y,x);
                double[] old = ci[1].get(y,x);

                for(int i = 0; i < curr.length; i++){
                    if(Math.abs(curr[i] - old[i]) > thresh){
                        ci[2].put(y,x,ci[3].get(y,x));
                    }
                }
            }, oldImg, oldestImg, actImg, rawestImg);

            long delta = (System.currentTimeMillis()-start);
            avg += (double)delta;
            avg /= 2.0;
            min = Math.min(delta, min);
            max = Math.max(delta, max);
            System.out.println("Took: "+delta+" Max: "+max+" Min: "+min+" Avg: "+avg);
            RoboGui.getInstance().show("Image",actImg);
        }
        capt.release();
    }
}
