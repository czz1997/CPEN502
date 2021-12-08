package NN;
import Activation.ReLU;
import Activation.Sigmoid;

public class QNet extends Module{

    public QNet(int in_dim, int h_dim){
        this.modules = new Object[]{
                new Linear(in_dim,h_dim),
                new ReLU(0.01),
                new Linear(h_dim,1),
                new Sigmoid(-2, 2)
        };
    }
}
