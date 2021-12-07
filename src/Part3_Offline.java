import Loss.LossBase;
import Loss.MSELoss;
import NN.QNet;
import Bot.State;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.Date;

public class Part3_Offline {
    public static void main(String[] args) {
        // load dataset and shuffle
        double[][] dataset_SA;
        double[][] dataset_Q;

        try {
            String datasetFileName = "dataset.txt";
            FileInputStream inputFile = new FileInputStream(datasetFileName);
            BufferedReader inputReader = new BufferedReader(new InputStreamReader(inputFile));

            int numRows = Integer.valueOf(inputReader.readLine());
            int numDimensions = Integer.valueOf(inputReader.readLine());

            dataset_SA = new double[numRows][numDimensions - 1];
            dataset_Q = new double[numRows][1];
            // load
            String[] rows = new String[numRows];
            for(int i = 0; i < numRows; i++) {
                rows[i] = inputReader.readLine();
            }
            // shuffle

            // fill
            for(int i = 0; i < numRows; i++){
                String[] vals = rows[i].split(",");
                for(int j = 0; j < numDimensions - 1; j++){
                    dataset_SA[i][j] = Double.parseDouble(vals[j]);
                }
                dataset_Q[i][0] = Double.parseDouble(vals[numDimensions - 1]);
            }
        }
        catch (IOException e){
            System.out.println("Dataset file read failed. Aborted.");
            return;
        }

        // hypers
        double lr = 0.0001;
        double momentum = 0.9;
        double targetError = 700;
        LossBase criterion = new MSELoss("half");

        // net
        int in_dim = State.length + 1;
        int h_dim = 32;
        QNet net = new QNet(in_dim, h_dim);


        // training
        net.initializeWeights();
        Trainer qNetTrainer = new Trainer(net, lr, momentum, criterion,
                dataset_SA, dataset_Q, targetError, true);
        int totalEpoch = qNetTrainer.train();
        System.out.println("Training completed in " + totalEpoch + "Epochs.");

        // save weights
        net.save(new File("out\\production\\CPEN502\\Bot\\RLBot.data\\" + (new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss")).format(new Date()) + ".weights"));
    }
}
