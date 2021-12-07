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

        // fetch non-zero q rows
        int rows = 0;
        for (int i=0;i<lut.getLength();i++) {
            int[] vector = lut.index2Vector(i);
            double[] sa = Arrays.stream(vector).asDoubleStream().toArray();
            double q = Math.max(-2, Math.min(2, lut.forward(sa)[0]));
            if(q != 0){
                rows ++;
            }
        }

        saveFile.println(rows);  // # of row
        saveFile.println(State.length + 2);  // # of dimensions per row

        for (int i=0;i<lut.getLength();i++) {
            StringBuilder row = new StringBuilder();
            int[] vector = lut.index2Vector(i);
            double[] sa = Arrays.stream(vector).asDoubleStream().toArray();
            double q = Math.max(-2, Math.min(2, lut.forward(sa)[0]));
            if(q == 0){
                continue;
            }
            for (int j=0; j<vector.length; j++) {
                double interval;
                if(j < State.length){
                    // state
                    interval = 2. / (State.upperBounds[j] - State.lowerBounds[j]);
                    // normalize
                    sa[j] = Math.min(1, Math.max(-1., -1. + (vector[j] - State.lowerBounds[j]) * interval));
                }
                else{
                    // action
                    interval = 2. / (Action.values().length - 1);
                    // normalize
                    sa[j] = Math.min(1., Math.max(-1., -1. + vector[j] * interval));
                }
                row.append(String.format("%f,", sa[j]));
            }
            row.append(String.format("%f", q));
            saveFile.println(row);
        }
        saveFile.close();
    }
}
