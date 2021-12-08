package Activation;

import java.util.Arrays;

public class NoActivation implements ActivationBase {
    private double[] output;
    @Override
    public double[] forward(double[] X) {
        double[] output = new double[X.length];
        // Element-wise
        System.arraycopy(X, 0, output, 0, X.length);
        // Cache output for backward
        this.output = output;
        return output;
    }

    @Override
    public double[] backward() {
        double[] grad = new double[this.output.length];
        Arrays.fill(grad, 1.);
        // clear output cache
        this.output = null;
        return grad;
    }
}
