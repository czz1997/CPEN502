package LUT;

import Bot.RLInterface;

import java.io.*;
import java.util.Arrays;

import robocode.*;

public class StateActionLUT implements LUTInterface, RLInterface {
    private final double[] LUT;
    private final int[] lowerBounds;
    private final int[] upperBounds;
    private final int[] coefficients;

    public StateActionLUT(int[] lowerBounds, int[] upperBounds){
        assert lowerBounds.length == upperBounds.length;
        this.coefficients = new int[lowerBounds.length];
        int i = 0;
        int product = 1;
        while(i<lowerBounds.length){
            this.coefficients[i] = product;
            product *= (upperBounds[i] - lowerBounds[i] + 1);
            i++;
        }
        System.out.println("LUT size " + product);
        this.LUT = new double[product];
        this.lowerBounds = lowerBounds;
        this.upperBounds = upperBounds;
        this.initialiseLUT();
    }
    public StateActionLUT(int[] lowerBounds, int[] upperBounds, String saveFileName) {
        this(lowerBounds, upperBounds);
        try{
            this.load(saveFileName);
        }
        catch (Exception e){
            e.printStackTrace();
        }
    }

    @Override
    public void updateQ(double[] stateActionVector, double newQ) {
        LUT[this.indexFor(stateActionVector)] = newQ;
    }

    @Override
    public double[] forward(double[] X) {
        double[] q = new double[1];
        q[0] = LUT[this.indexFor(X)];
        return q;
    }

    @Override
    public void save(File argFile) {
        PrintStream saveFile = null;

        try{
            saveFile = new PrintStream(new RobocodeFileOutputStream(argFile));
        } catch (IOException e) {
            System.out.println("[ERROR] Unable to create output stream for LUT.");
            return;
        }

        saveFile.println(this.LUT.length);  // # of row
        saveFile.println(1);  // # of dimensions per row

        for (double v : this.LUT) {
//            StringBuilder row = new StringBuilder();
//            int[] vector = this.index2Vector(i);
//            for (int k : vector) {
//                row.append(String.format("%d,", k));
//            }
            String row = String.format("%.2f", v);
            saveFile.println(row);
        }
        saveFile.close();
    }

    @Override
    public void load(String argFileName) throws IOException {
        FileInputStream inputFile = new FileInputStream(argFileName);
        BufferedReader inputReader = new BufferedReader(new InputStreamReader(inputFile));

        int numRows = Integer.valueOf(inputReader.readLine());
        int numDimensions = Integer.valueOf(inputReader.readLine());

        if(numRows != this.LUT.length || numDimensions != 1){
            System.out.printf("[ERROR] Dimension mismatch when loading LUT table from file: expected %s, %s, got %s, %s.\n",
                    this.LUT.length, 1, numRows, numDimensions);
            inputReader.close();
            throw new IOException();
        }
        for(int i=0;i<this.LUT.length;i++){
            String q = inputReader.readLine();
            this.LUT[i] = Double.parseDouble(q);
        }
    }

    @Override
    public void initialiseLUT() {
        Arrays.fill(this.LUT, 0);
    }

    @Override
    public int indexFor(double[] X) {
        assert X.length == this.lowerBounds.length;
        int index = 0;
        for(int i = 0; i<X.length; i++){
            if((int) X[i] < this.lowerBounds[i] || (int) X[i] > this.upperBounds[i]) {
                System.out.println(Arrays.toString(X));
                throw new IndexOutOfBoundsException();
            }
            index += (int) (X[i] - this.lowerBounds[i]) * this.coefficients[i];
        }
        return index;
    }

    public int[] index2Vector(int index){
        int[] vector = new int[this.coefficients.length];
        if(index >= this.LUT.length)
            throw new IndexOutOfBoundsException();
        for(int i = this.coefficients.length - 1; i>=0; i--){
            vector[i] = index / this.coefficients[i];
            index = index % this.coefficients[i];
        }
        return vector;
    }

    public int getLength(){
        return this.LUT.length;
    }
}
