package NN;

import Common.CommonInterface;

import java.util.HashMap;
import java.util.Map;

public interface NeuralNetInterface extends CommonInterface {
    /**
     * Get the gradients of current pass
     * @return gradient
     */
    public double[] grad();

    /**
     * Initialize weights to range (-0.5, 0.5)
     */
    public void initializeWeights();

    /**
     * Compute error signals
     * @param varargs {derivative of activation, signals} or {derivative of activation, signals, weights}
     */
    public void backward(Object... varargs);

    /**
     * Set weights to given weights values
     * @param state_dict saved weights values
     */
    public void load_state_dict(Map<Integer, Object> state_dict);

    /**
     * Update weights by applying gradient descend with given learning rate
     * @param lr learning rate
     */
    public void step(double lr);

    /**
     * Update weights by applying gradient descend with given learning rate and momentum
     * @param lr learning rate
     * @param momentum momentum
     */
    public void step(double lr, double momentum);

    /**
     * Get weights of the neurons
     * @return weights
     */
    public double[][] state_dict();
}
