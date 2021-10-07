import Loss.MSELoss;
import NN.XOR;
import java.security.InvalidParameterException;
import java.util.HashMap;
import java.util.Map;

public class Part1a {
    public static void main(String[] args) {
        // prepare dataset
        double[][] datasetX_binary = new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[][] datasetY_binary = new double[][]{{0}, {1}, {1}, {0}};
        double[][] datasetX_bipolar = new double[][]{{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
        double[][] datasetY_bipolar = new double[][]{{-1}, {1}, {1}, {-1}};
        // test weights
        Map<Integer, Object> state_dict = new HashMap<>();
        Map<Integer, Object> state_dict1 = new HashMap<>();
        state_dict1.put(0, new double[][]{
                {-0.3378,0.1970,0.3099},
                {0.2771,0.3191,0.1904},
                {0.2859,-0.1448,-0.0347},
                {-0.3329,0.3594,-0.4861}});
        Map<Integer, Object> state_dict2 = new HashMap<>();
        state_dict2.put(0, new double[][]{
                {-0.1401,0.4919,-0.2913,-0.3979,0.3581}});
        state_dict.put(0, state_dict1);
        state_dict.put(1, state_dict2);

        // switch Q
        String q = args[0];
        switch (q){
            case "a":{
                XOR xorNet = new XOR();
                xorNet.initializeWeights();
                Trainer xorTrainer = new Trainer(xorNet, 0.2, 0.0, new MSELoss("half"),
                        datasetX_binary, datasetY_binary, 0.05);
                xorTrainer.train();
            }
            break;
            case "b":{
                XOR xorNet = new XOR(true);
                xorNet.initializeWeights();
                Trainer xorTrainer = new Trainer(xorNet, 0.2, 0.0, new MSELoss("half"),
                        datasetX_bipolar, datasetY_bipolar, 0.05);
                xorTrainer.train();
            }
            break;
            case "c":{
                XOR xorNet = new XOR(true);
                xorNet.initializeWeights();
                Trainer xorTrainer = new Trainer(xorNet, 0.2, 0.9, new MSELoss("half"),
                        datasetX_bipolar, datasetY_bipolar, 0.05);
                xorTrainer.train();
            }
            break;
            case "testBinary":{
                // test binary
                XOR xorNet = new XOR();
                xorNet.load_state_dict(state_dict);  // load weights
                Trainer xorTrainer = new Trainer(xorNet, 0.2, 0.0, new MSELoss("sum"),
                        datasetX_binary, datasetY_binary, 0.05);
                xorTrainer.train();
            }
            break;
            case "testBipolar":{
                // test bipolar
                XOR xorNet = new XOR(true);
                xorNet.load_state_dict(state_dict);  // load weights
                Trainer xorTrainer = new Trainer(xorNet, 0.2, 0.0, new MSELoss("sum"),
                        datasetX_bipolar, datasetY_bipolar, 0.05);
                xorTrainer.train();
            }
            break;
            case "testBipolarMomentum":{
                // test bipolar
                XOR xorNet = new XOR(true);
                xorNet.load_state_dict(state_dict);  // load weights
                Trainer xorTrainer = new Trainer(xorNet, 0.2, 0.9, new MSELoss("sum"),
                        datasetX_bipolar, datasetY_bipolar, 0.05);
                xorTrainer.train();
            }
            break;
            default:
                throw new InvalidParameterException();
        }
    }
}