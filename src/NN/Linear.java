package NN;

import java.io.File;
import java.io.IOException;
import java.security.InvalidParameterException;
import java.util.Arrays;
import java.util.Map;

public class Linear implements NeuralNetInterface {
    // NN essentials
    private double[][] weights;
    private final int in_features;
    private final int out_features;
    // BP essentials
    private double[] gradient;
    private double[] input;
    private double[][] prev_delta;

    public Linear(int in_features, int out_features) {
        this.in_features = in_features;
        this.out_features = out_features;
        this.weights = new double[this.out_features][this.in_features + 1];
        this.prev_delta = new double[this.out_features][this.in_features + 1]; // same size as weights
        for (int i = 0; i < this.out_features; i++)
            for (int j = 0; j < this.in_features + 1; j++) {
                this.prev_delta[i][j] = 0; // previous weight change is 0
            }
        this.initializeWeights();
    }

    @Override
    public double[] grad() {
        return this.gradient;
    }

    @Override
    public void initializeWeights() {
        for (int i = 0; i < this.weights.length; i++)
            for (int j = 0; j < this.weights[0].length; j++) {
                this.weights[i][j] = Math.random() - 0.5; // initialize weights to range (-0.5, 0.5)
            }
    }

    @Override
    public double[] forward(double[] X) {
        assert (X.length == this.in_features) : "The size of input X (" + X.length + ")" +
                "must match the size of weights (" + this.in_features + ") at dimension 0";
        // Construct new X by insert bias at index 0
        double[] input = new double[this.in_features + 1];
        System.arraycopy(X, 0, input, 1, this.in_features);
        input[0] = 1.0;
        this.input = input; // cache input for weight update

        double[] output = new double[this.out_features];
        // Wx
        for (int i = 0; i < this.out_features; i++) {
            output[i] = 0;
            for (int j = 0; j < this.in_features + 1; j++) {
                output[i] += this.weights[i][j] * input[j];
            }
        }
        return output;
    }

    @Override
    public void save(File argFile) {

    }

    @Override
    public void load(String argFileName) throws IOException {

    }

    @Override
    public void load_state_dict(Map<Integer, Object> state_dict){
        double[][] weights = (double[][])state_dict.get(0);
        assert weights.length == this.weights.length;
        assert weights[0].length == this.weights[0].length;
        this.weights = Arrays.stream(weights).map(double[]::clone).toArray(double[][]::new);
    }

    @Override
    public void backward(Object... varargs) {
        switch (varargs.length){
            case 2:{
                // output layer
                double[] derivative = (double[])varargs[0];
                double[] signal = (double[])varargs[1];

                assert signal.length == this.out_features;
                assert derivative.length == this.out_features;
                this.gradient = signal;
                for(int i=0;i<this.out_features;i++){
                    this.gradient[i] *= derivative[i];
                }
            }
            break;
            case 3:{
                // hidden layer
                double[] derivative = (double[])varargs[0];
                double[] signal = (double[])varargs[1];
                double[][] weights = (double[][])varargs[2];

                assert weights[0].length == this.out_features + 1;
                assert signal.length == weights.length;
                assert derivative.length == this.out_features;
                this.gradient = new double[this.out_features];
                // Wt * signal
                for (int i = 0; i < this.out_features; i++) {
                    this.gradient[i] = 0;
                    for (int j = 0; j < signal.length; j++) {
                        this.gradient[i] += derivative[i] * signal[j] * weights[j][i + 1];  // bias is at index 0
                    }
                }
            }
            break;
            default: {
                throw new InvalidParameterException();
            }
        }
    }

    @Override
    public void step(double lr) {
        this.step(lr, 0.0);
    }

    @Override
    public void step(double lr, double momentum) {
        double[][] delta = new double[this.gradient.length][this.input.length];
        // signal * xt (outer product)
        for (int i = 0; i < this.gradient.length; i++) {
            for (int j = 0; j < this.input.length; j++) {
                delta[i][j] = momentum * this.prev_delta[i][j] + lr * this.gradient[i] * this.input[j];
            }
        }
        // update weights
        for (int i = 0; i < this.gradient.length; i++) {
            for (int j = 0; j < this.input.length; j++) {
                this.weights[i][j] += delta[i][j];
            }
        }
        // cache weight change
        this.prev_delta = delta;
        // clear input cache
        this.input = null;
    }

    @Override
    public double[][] state_dict() {
        return this.weights;
    }
}
