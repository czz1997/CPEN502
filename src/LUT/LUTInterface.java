package LUT;

import Common.CommonInterface;

public interface LUTInterface extends CommonInterface {
    public void initialiseLUT();
    public int indexFor(double [] X);
    public int getLength();
}
