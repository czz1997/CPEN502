import Loss.LossBase;
import NN.NeuralNetInterface;

import java.util.Arrays;
import java.util.Random;

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
    // display
    boolean verbose;

    public Trainer(NeuralNetInterface nn, double lr, double momentum,
                   LossBase lossFunction, double[][] datasetX, double[][] datasetY,
                   double targetError, boolean verbose){
        this.nn = nn;
        this.lr = lr;
        this.momentum = momentum;
        this.lossFunction = lossFunction;
        this.datasetX = datasetX;
        this.datasetY = datasetY;
        this.targetError = targetError;
        this.verbose = verbose;
        if(this.verbose)
            System.out.println("Trainer configuration: lr = " + lr + ", momentum = " + momentum);
    }

    public int train(){
        double error;  // total error
        int epochCounter = 0;
        do {
            // train
            Random random = new Random();
            for(int i=0;i<this.datasetX.length;i++){
                int index = random.nextInt(datasetX.length);
                double[] y_hat = this.nn.forward(this.datasetX[index]);  // input forward
                double[] loss = new double[]{this.datasetY[index][0] - y_hat[0]};  // compute loss
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
            if(this.verbose)
                System.out.println((epochCounter + 1) + " epoch: " + error);
            // increment epoch counter
            epochCounter += 1;
        }while (error > this.targetError && epochCounter < 750);
        return epochCounter;
    }
}
