package robogui;

import org.opencv.core.Mat;

import javax.security.auth.callback.Callback;
import javax.swing.*;
import java.awt.event.WindowListener;
import java.util.HashMap;
import java.util.function.Function;

public class RoboGui {
    private static RoboGui ourInstance = new RoboGui();
    public static boolean debug = false;

    public static RoboGui getInstance() {
        return ourInstance;
    }

    private HashMap<String, ImagePanel> panels;

    private RoboGui() {
        panels = new HashMap<>();
    }

    public static void debug(String message){
        if(debug){
            System.out.println(message);
        }
    }

    /**
     * Changes the image of an ImagePanel.
     * If the panel name does not exist, a panel will be created.
     * @param name Name of the ImagePanel
     * @param m New Image to be shown
     */
    public void show(String name, Mat m){
        ImagePanel panel = panels.get(name);
        if(panel == null){
            createPanel(name);
        }
        panel.setFrameSize(m.width(), m.height());
        panel.setImage(m);
    }

    /**
     * Same as show(), but if the panel name does not exists,
     * no panel will be crated.
     * @param name Name of the ImagePanel
     * @param m
     * @return False if there is no panel
     */
    public boolean showExists(String name, Mat m){
        if(panels.get(name) == null)
        {
            return false;
        }
        show(name, m);
        return true;
    }

    /**
     * Closes all ImagePanels
     */
    public void close(){
        panels.forEach((s, p) -> p.close());
        debug("All ImagePanels closed");
        allClosed.run();
    }

    /**
     * Closes an ImagePanel.
     * If the panel name does not exist, a panel will be created.
     * @param name
     */
    public void close(String name){
        ImagePanel panel = panels.get(name);
        if(panel == null){
            return;
        }
        panels.remove(name);
        debug(String.format("ImagePanel %s removed", name));
        if(panels.size() == 0){
            allClosed.run();
        }
        panel.close();
        debug(String.format("ImagePanel %s closed", name));
    }

    /**
     * Returns an ImagePanel.
     * If the panel name does not exist, a panel will be created.
     * @param name Name of the ImagePanel
     * @return
     */
    public ImagePanel get(String name){
        ImagePanel panel = panels.get(name);
        if(panel == null){
            panel = createPanel(name);
        }
        return panel;
    }

    /**
     * Creates a new ImagePanel.
     * @param name Name of the new ImagePanel
     * @return
     */
    public ImagePanel createPanel(String name){
        ImagePanel panel = new ImagePanel(name);
        panels.put(name, panel);
        debug(String.format("ImagePanel %s created", name));
        return panel;
    }

    /**
     * Adds a WindowListener to a panel.
     * If the panel name does not exist, a panel will be created.
     * @param name Name of the ImagePanel
     * @param listener
     */
    public void addWindowListener(String name, WindowListener listener){
        ImagePanel panel = panels.get(name);
        if(panel == null){
            panel = createPanel(name);
        }
        panel.addWindowListener(listener);
        debug(String.format("WindowListener added to ImagePanel %s", name));
    }

    /**
     * Sets the size of a frame.
     * If the panel name does not exist, a panel will be created.
     * @param name Name of the ImagePanel
     * @param width
     * @param height
     */
    public void setFrameSize(String name, int width, int height){
        ImagePanel panel = panels.get(name);
        if(panel == null){
            panel = createPanel(name);
        }
        panel.setFrameSize(width, height);
        debug(String.format("Set size of ImagePanel %s to %d %d", name, width, height));
    }

    public void setAllClosed(Runnable runnable){
        allClosed = runnable;
    }

    private Runnable allClosed;
}
