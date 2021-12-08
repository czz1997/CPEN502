package Activation;

public class ReLU implements ActivationBase {
    private double[] output;
    private double slope;
    public ReLU(){
        this(0.);
    }
    public ReLU(double slope){
        this.slope = slope;
    }
    public double[] forward(double[] X) {
        double[] output = new double[X.length];
        // Element-wise
        for (int i = 0; i < X.length; i++) {
            if(X[i] >= 0){
                output[i] = X[i];
            }
            else{
                output[i] = this.slope * X[i];
            }
        }
        // Cache output for backward
        this.output = output;
        return output;
    }

    @Override
    public double[] backward() {
        double[] grad = new double[this.output.length];
        for (int i = 0; i < grad.length; i++) {
            grad[i] = this.output[i] >= 0 ? 1 : slope;
        }
        // clear output cache
        this.output = null;
        return grad;
    }
}
