import Loss.LossBase;
import Loss.SELoss;
import org.junit.Assert;
import org.junit.Test;

public class LossTester {

    @Test
    public void testSELoss(){
        LossBase lossFunction = new SELoss();
        double[] input = new double[]{0.5};
        double[] y = new double[]{1.0};
        double[] actualOutput = lossFunction.forward(input, y);
        double[] expectedOutput = new double[]{0.125};
        Assert.assertEquals(actualOutput[0], expectedOutput[0], 0.0001);
    }
}
