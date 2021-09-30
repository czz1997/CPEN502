package Loss;

public interface LossBase{
    /**
     * Compute loss based on actual output and expected output
     * @param x actual output
     * @param y expected output
     * @return loss
     */
    public double[] forward(double[] x, double[] y);
}
