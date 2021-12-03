package NN;

import Bot.Action;
import Bot.RLInterface;
import Bot.State;

import java.io.File;
import java.io.IOException;

public class NNWrapper implements RLInterface {
    // net
    private final QNet net;

    // hypers
    double lr = 0.00001;
    double momentum = 0.9;

    public NNWrapper(){
        // net
        int in_dim = State.length + 1;
        int h_dim = 32;
        this.net = new QNet(in_dim, h_dim);
        this.net.initializeWeights();
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
        double[] y_hat = this.net.forward(stateActionVector);  // input forward
        double[] loss = new double[]{newQ - y_hat[0]};  // compute loss
        this.net.backward((Object) loss, this.lr, this.momentum);  // loss backward
    }

    @Override
    public double[] forward(double[] X) {
        this.normalize_X(X);
        return this.net.forward(X);
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
                interval = 2. / (State.upperBounds[i] - State.lowerBounds[i] + 1);
            }
            else{
                // action
                interval = 2. / Action.values().length;
            }
            // normalize
            X[i] = -1 + 0.5 * interval + X[i] * interval;
        }
    }
}
