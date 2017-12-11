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

        RoboGui.getInstance().show("Video",video);
        RoboGui.getInstance().get("Video").setFrameSize(video.width(),video.height());
        RoboGui.getInstance().get("Video").addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosed(WindowEvent e) {
                super.windowClosed(e);
                running = false;
            }
        });
        RoboGui.getInstance().show("Faces",video);
        RoboGui.getInstance().get("Faces").setFrameSize(video.width(),video.height());


        CascadeClassifier classifier = new CascadeClassifier(
                Thread.currentThread().getContextClassLoader().getResource("haarcascade_frontalface_default.xml")
                        .getPath());


        Mat gray = new Mat();
        Imgproc.cvtColor(video,gray, Imgproc.COLOR_RGB2GRAY);


        while(running) {
            capt.read(video);
            RoboGui.getInstance().show("Video", video);
            MatOfRect faces = new MatOfRect();
            classifier.detectMultiScale(gray, faces);
            List<Rect> rects = faces.toList();
            rects.forEach((rect) -> {
                Imgproc.rectangle(
                        video,
                        new Point(rect.x, rect.y),
                        new Point(rect.x + rect.width, rect.y + rect.height), Scalar.all(0),
                        2
                );
            });
            RoboGui.getInstance().show("Faces", video);
        }
    }
}
