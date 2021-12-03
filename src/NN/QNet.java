package NN;
import Activation.Sigmoid;

public class QNet extends Module{

    public QNet(int in_dim, int h_dim){
        this.modules = new Object[]{
                new Linear(in_dim,h_dim),
                new Sigmoid(-1, 1),
                new Linear(h_dim,1),
                new Sigmoid(-2, 2)
        };
    }
}
