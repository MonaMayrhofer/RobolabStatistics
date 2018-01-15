import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

import java.util.List;

public class App {
    public static void main(String[] arr){
        Graph g = new Graph();

        Tensor t1 = Tensors.create(new int[][] {{1,2},{3,4}});
        Tensor t2 = Tensors.create(new int[][] {{2,2},{2,2}});

        g.opBuilder("Mul","TimesTwo").build();

        Session s = new Session(g);
        List<Tensor<?>> out = s.runner().fetch("TimesTwo").run();
    }
}
