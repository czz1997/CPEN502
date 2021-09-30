import Activation.Sigmoid;
import org.junit.Assert;
import org.junit.Test;

public class ActivationTester {
    double[] input = new double[]{0.5, -0.6};

    @Test
    public void Test() {
        Sigmoid sigmoid = new Sigmoid();
        double[] forwardOutput = sigmoid.forward(input);
        double[] expectedForwardOutput = new double[]{0.6224593312, 0.3543436938};
        for (int i = 0; i < forwardOutput.length; ++i) {
            Assert.assertEquals(forwardOutput[i], expectedForwardOutput[i],0.001);
        }
        double[] backwardOutput = sigmoid.backward();
        double[] expectedBackwardOutput = new double[]{expectedForwardOutput[0] * (1 - expectedForwardOutput[0]), expectedForwardOutput[1] * (1 - expectedForwardOutput[1])};
        for (int i = 0; i < backwardOutput.length; ++i) {
            Assert.assertEquals(backwardOutput[i], expectedBackwardOutput[i], 0.001);
        }
    }
}
