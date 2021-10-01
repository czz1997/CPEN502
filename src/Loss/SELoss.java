package Loss;

public class SELoss implements LossBase {
    @Override
    public double[] forward(double[] x, double[] y) {
        assert x.length == y.length;
        double loss = 0;
        for (int i = 0; i < x.length; i++)
            loss += Math.pow(x[i] - y[i], 2.0);
        return new double[]{loss};
    }
}
