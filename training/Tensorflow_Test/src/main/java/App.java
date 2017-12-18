import org.tensorflow.*;

public class App {
    public static void main(String[] args) throws Exception {
        Tensor<Double> node1 = Tensors.create(3.0);
        Tensor<Double> node2 = Tensors.create(4.0);
        System.out.println(node1);
        Graph g = new Graph();
        Session sess = new Session(g);
        Session.Run run = new Session.Run();
        g.opBuilder("Const", "MyConst")
        .setAttr("dtype", DataType.DOUBLE)
        .setAttr("value", node1)
        .build();
        Tensor node3 = sess.runner().fetch("MyConst").run().get(0);
        System.out.println(node3.doubleValue());
    }

}
