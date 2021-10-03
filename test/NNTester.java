import Activation.Sigmoid;
import Loss.MSELoss;
import NN.Linear;
import org.junit.Assert;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

public class NNTester {

    @Test
    public void testLinearForward(){
        double[] inputVector = {0.5, -0.6};
        double[][] testWeights = {{1.2, 0.04, -0.96}};
        Map<Integer, Object> state_dict = new HashMap<>();
        state_dict.put(0, testWeights);
        Linear linear = new Linear(2, 1);
        linear.load_state_dict(state_dict);
        double actualOutput = linear.forward(inputVector)[0];
        double expectedOutput = 1.796;

        Assert.assertEquals(expectedOutput, actualOutput, 0.000001);
    }

    @Test
    public void testLinearBackward(){
        double[] inputVector = {0.5, -0.6};
        double[][] testWeights = {{1.2, 0.04, -0.96}};
        Map<Integer, Object> state_dict = new HashMap<>();
        state_dict.put(0, testWeights);
        // linear
        Linear linear = new Linear(2, 1);
        linear.load_state_dict(state_dict);
        // sigmoid
        Sigmoid sigmoid = new Sigmoid();

        // forward
        double[] output = linear.forward(inputVector);  // 1.796
        output = sigmoid.forward(output);  // 0.8576613198

        // error
        MSELoss loss = new MSELoss("half");
        double[] error = loss.forward(output, new double[]{1.0});  // 0.01013015

        // backward
        double[] fs = sigmoid.backward();  // 0.1220783803
        linear.backward(fs, error);

        // grad check
        double[] expectedGrad = new double[]{0.0012366723};
        Assert.assertEquals(linear.grad()[0], expectedGrad[0],0.000001);

        // step
        linear.step(0.2, 0.0);

        // weights check
        double[][] actualWeights = linear.state_dict();
        double[][] expectedWeights = new double[][]{{1.2002473345,0.0401236672,-0.9601484007}};
        for(int i=0;i<3;i++){
            Assert.assertEquals(actualWeights[0][i], expectedWeights[0][i],0.000001);
        }
    }

    @Test
    public void testMLPBackward(){
        double[] inputVector = {0.5, -0.6};
        double[][] testWeights1 = {{1.2, 0.04, -0.96}, {1.2, 0.04, -0.96}};
        double[][] testWeights2 = {{1.2, 0.04, -0.96}};
        Map<Integer, Object> state_dict1 = new HashMap<>();
        state_dict1.put(0, testWeights1);
        Map<Integer, Object> state_dict2 = new HashMap<>();
        state_dict2.put(0, testWeights2);
        // linear
        Linear linear1 = new Linear(2, 2);
        linear1.load_state_dict(state_dict1);
        Linear linear2 = new Linear(2, 1);
        linear2.load_state_dict(state_dict2);
        // sigmoid
        Sigmoid sigmoid1 = new Sigmoid();
        Sigmoid sigmoid2 = new Sigmoid();

        // forward
        double[] output = linear1.forward(inputVector);  // 1.796, 1.796
        output = sigmoid1.forward(output);  // 0.8576613198, 0.8576613198
        output = linear2.forward(output);  // 0.4109515158
        output = sigmoid2.forward(output);  // 0.6013160125
        double actualOutput = output[0];
        double expectedOutput = 0.6013160125;
        Assert.assertEquals(expectedOutput, actualOutput, 0.000001);

        // error
        MSELoss loss = new MSELoss("half");
        double[] error = loss.forward(output, new double[]{1.0});  // 0.079474461

        // backward
        double[] fs = sigmoid2.backward();  //  0.2397350656
        linear2.backward(fs, error);  // signal 0.0190528151
        Assert.assertEquals(linear2.grad()[0], 0.0190528151,0.000001);
        fs = sigmoid1.backward();  // 0.1220783803, 0.1220783803
        linear1.backward(fs, linear2.grad(), linear2.state_dict());

        // grad check
        double[] expectedGrad = new double[]{0.0000930375, -0.0022328993};
        for(int i=0;i<2;i++)
            Assert.assertEquals(linear1.grad()[i], expectedGrad[i],0.000001);
    }
}
