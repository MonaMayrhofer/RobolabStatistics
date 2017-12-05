/*
Erik Mayrhofer - Proof Of Concept of Background-Reduction
 */
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import javax.swing.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

public class App {

    private static boolean running = true;

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) throws InterruptedException{
        int thresh = 3; //Play around with this value
        JFrame frame = new JFrame("TestJFrame");
        ImagePanel panel = new ImagePanel();
        frame.add(panel);
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);

        VideoCapture capt = new VideoCapture();
        capt.open(0);

        frame.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) { running = false; }
        });

        Mat currImg = new Mat();
        Mat rawestImg = new Mat();
        Mat rawImg = new Mat();
        Mat oldImg = new Mat();
        Mat oldestImg;

        capt.read(oldImg);
        Mat actImg = new Mat(oldImg.height(), oldImg.width(), oldImg.type(), Scalar.all(0));
        Imgproc.cvtColor(oldImg, oldImg, Imgproc.COLOR_RGB2GRAY);
        frame.setSize(oldImg.width(),oldImg.height());
        frame.setVisible(true);

        while(running){
            actImg.setTo(Scalar.all(255));
            capt.read(rawestImg);

            oldestImg = oldImg.clone();

            Imgproc.cvtColor(rawestImg, rawImg, Imgproc.COLOR_RGB2GRAY);
            Core.multiply(oldImg, Scalar.all(0.9), oldImg);
            Core.multiply(rawImg, Scalar.all(0.1), currImg);
            Core.add(currImg, oldImg, oldImg);

            long start = System.currentTimeMillis();

            ConcurrentMatMath m = new ConcurrentMatMath(3);

            m.mutateMat((x, y, ci) -> {
                double[] curr = ci[0].get(y,x);
                double[] old = ci[1].get(y,x);

                for(int i = 0; i < curr.length; i++){
                    if(Math.abs(curr[i] - old[i]) > thresh){
                        ci[2].put(y,x,ci[3].get(y,x));
                    }
                }
            }, oldImg, oldestImg, actImg, rawestImg);

            /*
            for(int x = 0; x < currImg.width(); x++){
                for(int y = 0; y < currImg.height(); y++){
                    double[] curr = oldImg.get(y,x);
                    double[] old = oldestImg.get(y,x);

                    for(int i = 0; i < curr.length; i++){
                        if(Math.abs(curr[i] - old[i]) > thresh){
                            actImg.put(y,x,rawestImg.get(y,x));
                        }
                    }
                }
            }*/


            System.out.println("Took: "+(System.currentTimeMillis()-start));
            panel.setImage(actImg);
        }
        capt.release();
    }
}
