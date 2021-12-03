package LUT;

import Bot.Action;
import Bot.State;
import robocode.RobocodeFileOutputStream;

import java.awt.*;
import java.io.*;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;

public class LUT2Dataset {
    public static void main(String[] args) {
        // load LUT
        StateActionLUT lut = new StateActionLUT(State.lowerBounds, State.upperBounds);
        try{
            lut.load("out\\production\\CPEN502\\Bot\\RLBot.data\\LUT-35000.txt");
        }
        catch (Exception e){
            System.out.println("Couldn't load LUT table.");
        }

        // create output file
        PrintStream saveFile = null;

        try{
            saveFile = new PrintStream(new FileOutputStream((new File("dataset.txt")).getAbsolutePath()));
        } catch (IOException e) {
            System.out.println("[ERROR] Unable to create dataset file.");
            return;
        }

        saveFile.println(lut.getLength());  // # of row
        saveFile.println(State.length + 2);  // # of dimensions per row

        for (int i=0;i<lut.getLength();i++) {
            StringBuilder row = new StringBuilder();
            int[] vector = lut.index2Vector(i);
            double[] sa = Arrays.stream(vector).asDoubleStream().toArray();
            double q = Math.max(-1, Math.min(1, lut.forward(sa)[0]));
            for (int j=0; j<vector.length; j++) {
                double interval;
                if(j < State.length){
                    // state
                    interval = 2. / (State.upperBounds[j] - State.lowerBounds[j] + 1);
                }
                else{
                    // action
                    interval = 2. / Action.values().length;
                }
                // normalize
                sa[j] = -1 + 0.5 * interval + vector[j] * interval;
                row.append(String.format("%.2f,", sa[j]));
            }
            row.append(String.format("%.2f", q));
            saveFile.println(row);
        }
        saveFile.close();
    }
}
