import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;
import robogui.RoboGui;

import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.*;
import java.net.URL;
import java.util.List;

public class App {

    private static boolean running = true;
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args){
        VideoCapture capt = new VideoCapture();
        capt.open(0);

        Mat video = new Mat();
        capt.read(video);

        RoboGui.getInstance().show("Faces",video);
        RoboGui.getInstance().get("Faces").addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosed(WindowEvent e) {
                super.windowClosed(e);
                running = false;
                RoboGui.getInstance().close();
            }
        });
        RoboGui.getInstance().get("Faces").setFrameSize(video.width(),video.height());


        CascadeClassifier classifier = new CascadeClassifier(
                Thread.currentThread().getContextClassLoader().getResource("haarcascade_frontalface_default.xml")
                        .getPath());


        Mat gray = new Mat();
        Imgproc.cvtColor(video,gray, Imgproc.COLOR_RGB2GRAY);


        Mat heatMap = new Mat( video.width(), video.height(), CvType.CV_32FC1, Scalar.all(0));

        while(running) {
            capt.read(video);
            Imgproc.cvtColor(video,gray, Imgproc.COLOR_RGB2GRAY);
            RoboGui.getInstance().show("Gray",gray);

            MatOfRect faces = new MatOfRect();
            classifier.detectMultiScale(gray, faces);
            List<Rect> rects = faces.toList();
            rects.forEach((rect) -> {
                Imgproc.rectangle(
                        heatMap,
                        new Point(rect.x, rect.y),
                        new Point(rect.x + rect.width, rect.y + rect.height), Scalar.all(0),
                        2
                );
            });
            RoboGui.getInstance().show("Faces", video);
        }
    }
}
