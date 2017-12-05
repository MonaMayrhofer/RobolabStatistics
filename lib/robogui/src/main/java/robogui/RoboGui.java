package robogui;/*
Mat m;

--Kein Fenster offen

robogui.RoboGui.gi().show("F1", m);
--Fenster öffnet sich und zeigt m an

Mat g;
robogui.RoboGui.gi().show("F1", g);
--Selbes fenster zeigt jetzt g an

robogui.RoboGui.gi().show("F2", m);
--Neues Fenster geht jetzt auf mit f2

robogui.RoboGui.gi().close();

robogui.RoboGui.gi().get("F1");
--gibt f zrück

 */

import org.opencv.core.Mat;

import java.awt.event.WindowListener;
import java.util.HashMap;

public class RoboGui {
    private static RoboGui ourInstance = new RoboGui();

    public static RoboGui getInstance() {
        return ourInstance;
    }

    private HashMap<String, ImagePanel> panels;

    private RoboGui() {
        panels = new HashMap<>();
    }

    public void show(String name, Mat m){
        ImagePanel panel = panels.get(name);
        if(panel == null){
            panel = createPanel(name);
        }
        panel.setImage(m);
    }

    public void close(){
        panels.forEach((s, p) -> p.close());
    }

    public void close(String name){
        ImagePanel panel = panels.get(name);
        if(panel == null){
            return;
        }
        panels.remove(panel);
        panel.close();
    }

    public ImagePanel get(String name){
        ImagePanel panel = panels.get(name);
        if(panel == null){
            panel = createPanel(name);
        }
        return panel;
    }

    public ImagePanel createPanel(String name){
        ImagePanel panel = new ImagePanel(name);
        panels.put(name, panel);
        System.out.println("Panel created");
        return panel;
    }

    public void addWindowListener(String name, WindowListener listener){
        ImagePanel panel = panels.get(name);
        if(panel == null){
            panel = createPanel(name);
        }
        panel.addWindowListener(listener);
        System.out.println("WindowListener added");
    }

    public void setFrameSize(String name, int width, int height){
        ImagePanel panel = panels.get(name);
        if(panel == null){
            panel = createPanel(name);
        }
        panel.setFrameSize(width, height);
        System.out.println("FrameSize set");
    }
}
