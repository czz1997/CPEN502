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
    double lr = 0.00001;
    double momentum = 0.9;
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
        this.normalize_X(stateActionVector);
        if(this.buffer_enabled){
            // buffer experience
            double[] experience = new double[stateActionVector.length + 1];
            System.arraycopy(stateActionVector, 0, experience, 0, stateActionVector.length);
            experience[experience.length - 1] = newQ;
            this.buffer.add(experience);
            // draw experiences
            Object[] experiences = this.buffer.randomSample(replay_sample_size);
            double[] X = new double[stateActionVector.length];
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
            double[] y_hat = this.net.forward(stateActionVector);  // input forward
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
        for (int i=0; i<X.length; i++) {
            double interval;
            if(i < State.length){
                // state
                interval = 2. / (State.upperBounds[i] - State.lowerBounds[i]);
                // normalize
                X[i] = Math.min(1, Math.max(-1., -1. + (X[i] - State.lowerBounds[i]) * interval));
            }
            else{
                // action
                interval = 2. / (Action.values().length - 1);
                // normalize
                X[i] = Math.min(1., Math.max(-1., -1. + X[i] * interval));
            }
        }
    }
}
