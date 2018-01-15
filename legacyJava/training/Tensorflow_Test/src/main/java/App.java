import org.tensorflow.*;

import java.nio.DoubleBuffer;

public class App {
    public static void main(String[] args) throws Exception {
        Tensor<Double> aT = Tensors.create(new double[]{3.0, 9.0});
        Tensor<Double> bT = Tensors.create(new double[]{4.0, 5.0});
        System.out.println(aT);
        System.out.println(bT);
        Graph g = new Graph();
        Session sess = new Session(g);
        Output<Double> node1 = g.opBuilder("Placeholder", "a")
                .setAttr("dtype", DataType.DOUBLE)
                .build()
                .output(0);
        Output<Double> node2 = g.opBuilder("Placeholder", "b")
                .setAttr("dtype", DataType.DOUBLE)
                .build()
                .output(0);
        Output<Double> node3 = g.opBuilder("Mul", "MyMul")
                .addInput(node1)
                .addInput(node2)
                .build()
                .output(0);
        Tensor<Double> result = sess.runner().feed("a", aT).feed("b", bT).fetch("MyMul").run().get(0).expect(Double.class);
        System.out.println(result);
        Tensor<Double> cT = Tensors.create(new double[]{5.0, 2.0});
        Output<Double> node4 = g.opBuilder("Placeholder", "c")
                .setAttr("dtype", DataType.DOUBLE)
                .build()
                .output(0);
        g.opBuilder("Add", "MulAdd")
                .addInput(node3)
                .addInput(node4)
                .build();
        result = sess.runner().feed("a", aT).feed("b", bT).feed("c", cT).fetch("MulAdd").run().get(0).expect(Double.class);
        System.out.println(result);

        Output<Double> var1 = g.opBuilder("Variable", "mulVar")
                .setAttr("dtype", DataType.DOUBLE)
                .setAttr("shape", Shape.make(1))
                .build()
                .output(0);
        var1 = g.opBuilder("Assign", "mulVarAssign")
                .addInput(var1)
                .addInput(node3)
                .build()
                .output(0);
        g.opBuilder("Add", "VarAdd")
                .addInput(node3)
                .addInput(var1)
                .build();
        result = sess.runner().feed("a", aT).feed("b", bT).feed("c", cT).fetch("VarAdd").run().get(0).expect(Double.class);
        DoubleBuffer resultB = DoubleBuffer.wrap(new double[]{0.0, 0.0});
        result.writeTo(resultB);
        System.out.println(resultB.get(0));
        System.out.println(resultB.get(1));
    }
}
