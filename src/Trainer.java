import Loss.LossBase;
import NN.NeuralNetInterface;

import java.util.Arrays;

public class Trainer {
    // network
    NeuralNetInterface nn;
    // learning configuration
    double lr;
    double momentum;
    // dataset
    double[][] datasetX;
    double[][] datasetY;
    // evaluation
    LossBase lossFunction;
    double targetError;

    public Trainer(NeuralNetInterface nn, double lr, double momentum,
                   LossBase lossFunction, double[][] datasetX, double[][] datasetY,
                   double targetError){
        this.nn = nn;
        this.lr = lr;
        this.momentum = momentum;
        this.lossFunction = lossFunction;
        this.datasetX = datasetX;
        this.datasetY = datasetY;
        this.targetError = targetError;
        System.out.println("Trainer configuration: lr = " + lr + ", momentum = " + momentum);
    }

    public void train(){
        double error;  // total error
        int epochCounter = 0;
        do {
            // train
            for(int i=0;i<this.datasetX.length;i++){
                double[] y_hat = this.nn.forward(this.datasetX[i]);  // input forward
                double[] loss = new double[]{this.datasetY[i][0] - y_hat[0]};  // compute loss
                this.nn.backward((Object) loss, this.lr, this.momentum);  // loss backward
//                this.nn.step(this.lr, this.momentum);  // weight step
            }
            // compute error sum
            error = 0;
            for(int i=0;i<this.datasetX.length;i++){
                double[] y_hat = this.nn.forward(this.datasetX[i]);  // input forward
                double[] loss = lossFunction.forward(this.datasetY[i], y_hat);  // compute loss
                error += Arrays.stream(loss).sum();
            }
            System.out.println((epochCounter + 1) + " epoch: " + error);
            // increment epoch counter
            epochCounter += 1;
        }while (error > this.targetError);
    }
}
