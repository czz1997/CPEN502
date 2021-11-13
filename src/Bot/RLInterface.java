package Bot;

import Common.CommonInterface;

public interface RLInterface extends CommonInterface {
    public void updateQ(double[] stateActionVector, double newQ);
}
