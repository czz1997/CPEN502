import Loss.LossBase;
import Loss.SELoss;
import NN.XOR;
import com.sun.javaws.exceptions.InvalidArgumentException;

import java.security.InvalidParameterException;

public class Part1a {
    public static void main(String[] args) {
        // prepare dataset
        double[][] datasetX = new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[][] datasetY = new double[][]{{0}, {1}, {1}, {0}};

        // switch Q
        String q = args[0];
        switch (q){
            case "a":{
                XOR xorNet = new XOR();
                xorNet.initializeWeights();
                Trainer xorTrainer = new Trainer(xorNet, 0.2, 0.0, new SELoss(),
                        datasetX, datasetY, 0.05);
                xorTrainer.train();
            }
            break;
            case "b":{
                XOR xorNet = new XOR(true);
                xorNet.initializeWeights();
                Trainer xorTrainer = new Trainer(xorNet, 0.2, 0.0, new SELoss(),
                        datasetX, datasetY, 0.05);
                xorTrainer.train();
            }
            case "c":{
                XOR xorNet = new XOR(true);
                xorNet.initializeWeights();
                Trainer xorTrainer = new Trainer(xorNet, 0.2, 0.9, new SELoss(),
                        datasetX, datasetY, 0.05);
                xorTrainer.train();
            }
            break;
            case "testBinary":{
                // test binary
                XOR xorNet = new XOR();
                xorNet.load_state_dict(new double[0][0]);  // load weights
                Trainer xorTrainer = new Trainer(xorNet, 0.02, 0.0, new SELoss(),
                        datasetX, datasetY, 0.05);
                xorTrainer.train();
            }
            break;
            case "testBipolar":{
                // test bipolar
                XOR xorNet = new XOR(true);
                xorNet.load_state_dict(new double[0][0]);  // load weights
                Trainer xorTrainer = new Trainer(xorNet, 0.02, 0.0, new SELoss(),
                        datasetX, datasetY, 0.05);
                xorTrainer.train();
            }
            break;
            default:
                throw new InvalidParameterException();
        }
    }
}