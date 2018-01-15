package robogui;

import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.WindowListener;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;

public class ImagePanel extends JPanel {

    private Mat image;
    private JFrame frame;
    private int width;
    private int height;

    public synchronized Mat getImage() {
        return image;
    }

    public synchronized void setImage(Mat image) {
        this.image = image.clone();
        repaint();
    }

    public ImagePanel(String name){
        frame = new JFrame(name);
        frame.add(this);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
    }

    /**
     * Closes the JPanel.
     */
    public void close(){
        frame.dispose();
    }

    /**
     * Adds a WindowListener to the JPanel.
     * @param listener
     */
    public void addWindowListener(WindowListener listener){
        frame.addWindowListener(listener);
    }

    /**
     * Sets the size of the JPanel.
     * @param width
     * @param height
     */
    public void setFrameSize(int width, int height){
        if(this.width != width || this.height != height){
            this.width = width;
            this.height = height;
            frame.setSize(width, height);
        }
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
    }
}
