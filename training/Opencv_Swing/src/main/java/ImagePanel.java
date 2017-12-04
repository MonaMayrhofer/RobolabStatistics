import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;

public class ImagePanel extends JPanel {

    private Mat image;

    public synchronized Mat getImage() {
        return image;
    }

    public synchronized void setImage(Mat image) {
        this.image = image.clone();
    }

    @Override
    public void paintComponent(Graphics g) {
        super.paintComponents(g);

        if(getImage() == null) return;

        MatOfByte matOfByte = new MatOfByte();

        Imgcodecs.imencode(".jpg", getImage(), matOfByte);

        byte[] byteArray = matOfByte.toArray();
        BufferedImage bufImage = null;
        try {

            InputStream in = new ByteArrayInputStream(byteArray);
            bufImage = ImageIO.read(in);
            g.drawImage(bufImage,0,0,null);
        } catch (Exception e) {
            e.printStackTrace();
        }

        repaint();
    }
}
