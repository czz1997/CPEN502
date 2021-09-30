package Common;

import java.io.File;
import java.io.IOException;

public interface CommonInterface {
    public double[] forward(double [] X);
    public void save(File argFile);
    public void load(String argFileName) throws IOException;
}
