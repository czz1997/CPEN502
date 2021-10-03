package Loss;

import com.sun.javaws.exceptions.InvalidArgumentException;

import java.security.InvalidParameterException;
import java.util.Objects;

public class MSELoss implements LossBase {
    private String reduction = "mean";

    public MSELoss(){
        this("mean");
    }

    public MSELoss(String reduction){
        if(!Objects.equals(reduction, "mean") &&
                !Objects.equals(reduction, "sum") &&
                !Objects.equals(reduction, "half")){
            throw new InvalidParameterException("Reduction \"" + reduction + "\" is invalid.");
        }
        this.reduction = reduction;
    }

    @Override
    public double[] forward(double[] x, double[] y) {
        assert x.length == y.length;
        double loss = 0;
        for (int i = 0; i < x.length; i++)
            loss += Math.pow(x[i] - y[i], 2.0);
        switch (this.reduction){
            case "sum":
            case "half":
                loss *= 0.5;
                break;
            default:
                loss /= x.length;
        }
        return new double[]{loss};
    }
}
