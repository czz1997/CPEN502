package NN;

import Activation.ActivationBase;
import Activation.Sigmoid;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class XOR implements NeuralNetInterface{

    public Object[] modules;

    public XOR(){
        this(false);
    }

    public XOR(boolean bipolar){
        this.modules = new Object[]{
                new Linear(2,4),
                bipolar?new Sigmoid(-1, 1): new Sigmoid(),
                new Linear(4,1),
                bipolar?new Sigmoid(-1, 1): new Sigmoid()};
    }

    @Override
    public double[] forward(double[] X) {
        double[] output = X;
        for(Object module: this.modules){
            if(module instanceof NeuralNetInterface){
                output = ((NeuralNetInterface) module).forward(output);
            }
            else if(module instanceof ActivationBase){
                output = ((ActivationBase) module).forward(output);
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
    public double[] grad() {
        return new double[0];
    }

    @Override
    public void initializeWeights() {
        for(Object module: this.modules){
            if(module instanceof NeuralNetInterface){
                ((NeuralNetInterface) module).initializeWeights();
            }
        }
    }

    @Override
    // signal is loss
    public void backward(Object... varargs) {
        assert varargs.length == 3;

        double lr = (double) varargs[1];
        double momentum = (double) varargs[2];

        double[] signal = (double[]) varargs[0];  // error signal
        boolean output = true;  // flag to indicate output layer
        double[] fs = new double[0];  // temp var for derivative of activation
        double[][] weights = new double[0][0];  // temp var for previous weights

        for(int i=this.modules.length - 1;i>=0;i--){
            if(this.modules[i] instanceof ActivationBase){
                fs = ((ActivationBase) this.modules[i]).backward();
            }
            else if(this.modules[i] instanceof NeuralNetInterface){
                NeuralNetInterface nn = (NeuralNetInterface) this.modules[i];
                if(output)
                {
                    output = false;
                    nn.backward(fs, signal);
                }
                else {
                    nn.backward(fs, signal, weights);
                }
                nn.step(lr, momentum);
                signal = nn.grad();
                weights = nn.state_dict();
            }
        }
    }

    @Override
    public void load_state_dict(Map<Integer, Object> state_dict) {
        int keyCounter = 0;
        for(Object module: this.modules){
            if(module instanceof NeuralNetInterface){
                Map<Integer, Object> sub_state_dict = (Map<Integer, Object>)(state_dict.get(keyCounter));
                ((NeuralNetInterface) module).load_state_dict(sub_state_dict);
                keyCounter += 1;
            }
        }
    }

    @Override
    public void step(double lr) {
        this.step(lr, 0);
    }

    @Override
    public void step(double lr, double momentum) {
//        for(Object module: this.modules){
//            if(module instanceof NN.NeuralNetInterface){
//                ((NN.NeuralNetInterface) module).step(lr, momentum);
//            }
//        }
    }

    @Override
    public double[][] state_dict() {
        return new double[0][];
    }
}
