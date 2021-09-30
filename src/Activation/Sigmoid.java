package Activation;

public class Sigmoid implements ActivationBase {

    private final int a;
    private final int b;
    private double[] output;

    // custom sigmoid
    public Sigmoid(int a, int b) {
        this.a = a;
        this.b = b;
    }

    // bipolar sigmoid
    public Sigmoid(boolean bipolar){
        this(bipolar?-1:0, 1);
    }

    // binary sigmoid
    public Sigmoid() {
        this(0, 1);
    }

    @Override
    public double[] forward(double[] X) {
        double[] output = new double[X.length];
        // Element-wise
        for (int i = 0; i < X.length; i++) {
            output[i] = (this.b - this.a) / (1 + Math.pow(Math.E, -X[i])) - (-this.a);
        }
        // Cache output for backward
        this.output = output;
        return output;
    }

    @Override
    public double[] backward() {
        double[] grad = new double[this.output.length];
        for (int i = 0; i < grad.length; i++) {
            grad[i] = 1.0 / (this.b - this.a) * (-this.b * this.a + (this.b + this.a) * this.output[i] - Math.pow(this.output[i], 2.0));
        }
        // clear output cache
        this.output = null;
        return grad;
    }
}
