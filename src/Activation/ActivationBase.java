package Activation;

public interface ActivationBase{
    /**
     * Forward the input vector to the activation function
     * @param X input vector z
     * @return activated output vector a
     */
    public double[] forward(double[] X);

    /**
     * Compute activation derivative
     * @return derivative f'(z)
     */
    public double[] backward();
}
