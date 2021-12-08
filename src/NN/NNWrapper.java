package NN;

import Bot.Action;
import Bot.RLInterface;
import Bot.State;
import Utils.ReplayMemory;

import java.io.File;
import java.io.IOException;

public class NNWrapper implements RLInterface {
    // net
    private final QNet net;
    // replay buffer
    private final ReplayMemory<double[]> buffer;

    // hypers
    double lr = 0.0005;
    double momentum = 0.95;
    int buffer_size = 20000;
    boolean buffer_enabled = false;
    int replay_sample_size = 8;

    public NNWrapper(){
        // net
        int in_dim = State.length + 1;
        int h_dim = 32;
        this.net = new QNet(in_dim, h_dim);
        this.net.initializeWeights();
        // replay buffer
        buffer = new ReplayMemory<double[]>(buffer_size);
    }
    public NNWrapper(String weightFile){
        this();
        try{
            this.load(weightFile);
        }
        catch (IOException e){
            System.out.println("[ERROR] Unable to load weights.");
        }
    }

    @Override
    public void updateQ(double[] stateActionVector, double newQ) {
        newQ = Math.max(-2., Math.min(2., newQ));
        double[] input = new double[stateActionVector.length];
        System.arraycopy(stateActionVector, 0, input, 0, stateActionVector.length);
        this.normalize_X(input);
        if(this.buffer_enabled){
            // buffer experience
            double[] experience = new double[input.length + 1];
            System.arraycopy(input, 0, experience, 0, input.length);
            experience[experience.length - 1] = newQ;
            this.buffer.add(experience);
            // draw experiences
            Object[] experiences = this.buffer.randomSample(replay_sample_size);
            double[] X = new double[input.length];
            for(Object sample: experiences){
                double[] exp = (double[]) sample;
                System.arraycopy(exp, 0, X, 0, X.length);
                double Y = exp[exp.length-1];
                double[] y_hat = this.net.forward(X);  // input forward
                double[] loss = new double[]{Y - y_hat[0]};  // compute loss
                this.net.backward((Object) loss, this.lr, this.momentum);  // loss backward
            }
        }
        else{
            double[] y_hat = this.net.forward(input);  // input forward
            double[] loss = new double[]{newQ - y_hat[0]};  // compute loss
            this.net.backward((Object) loss, this.lr, this.momentum);  // loss backward
        }
    }

    @Override
    public double[] forward(double[] X) {
        double[] input = new double[X.length];
        System.arraycopy(X, 0, input, 0, X.length);
        this.normalize_X(input);
        return this.net.forward(input);
    }

    @Override
    public void save(File argFile) {
        this.net.save(argFile);
    }

    @Override
    public void load(String argFileName) throws IOException {
        this.net.load(argFileName);
    }

    private void normalize_X(double[] X){
        double[] lowerBounds = {0, 0, 0, 0, -180, 0, 0, 0};
        double[] upperBounds = {800, 600, 200, 360, 180, 1000, 200, 4};
        for (int i=0; i<X.length; i++) {
            double interval = 2. / (upperBounds[i] - lowerBounds[i]);;
            X[i] = -1. + (X[i] - lowerBounds[i]) * interval;
            if(X[i] < -1 && -1 - X[i] > 0.05 || X[i] > 1 && X[i] -1 > 0.05){
                System.out.println("[WARN] Input at index " + i + " at value " + X[i] + " out of range.");
            }
        }
    }
}
