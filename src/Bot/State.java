package Bot;

import LUT.LUTInterface;
import NN.NeuralNetInterface;

public class State {
    public static int length = 7;
    private double x;
    private double y;
    private double energy;
    private double heading;
    private double enemyBearing;
    private double enemyDistance;
    private double enemyEnergy;
    public static int[] lowerBounds = {0, 0, 0, 0, 0, 0, 0, 0};
    public static int[] upperBounds = {7, 5, 3, 3, 3, 3, 3, Action.values().length-1};
    // the following state(s) won't be in the state vector
    private double enemyHeading;
    private double enemyVelocity;
    private double velocity;
    private double gunHeading;
    // settings
    private boolean quantize;

    // constructor
    public State(RLInterface agent){
        // set to default
        if(agent instanceof LUTInterface){
            quantize = true;
        }
        else if(agent instanceof NeuralNetInterface){
            quantize = false;
        }
    }

    // setters
    public void setMyState(double x, double y,
                           double energy,
                           double heading, double gunHeading,
                           double velocity){
        this.x = x;
        this.y = y;
        this.energy = energy;
        this.heading = heading;
        this.gunHeading = gunHeading;
        this.velocity = velocity;
    }

    public void setEnemyState(double theta, double distance,
                              double enemyEnergy,
                              double enemyHeading,
                              double enemyVelocity){
        this.enemyBearing = theta;
        this.enemyDistance = distance;
        this.enemyEnergy = enemyEnergy;
        this.enemyHeading = enemyHeading;
        this.enemyVelocity = enemyVelocity;
    }

    // getters
    public double getHeading(){
        return this.heading;
    }

    public double getGunHeading(){
        return this.gunHeading;
    }

    public double getEnemyBearing(){
        return this.enemyBearing;
    }

    // state vector
    public double[] getStateVector(){
        if(this.quantize) {
            return new double[]{
                    quantizeX(this.x), quantizeY(this.y),
                    Energy.quantizeEnergy(this.energy).getValue(),
                    Heading.quantizeHeading(this.heading, this.x, this.y).getValue(),
                    Bearing.quantizeBearing(this.enemyBearing, this.x, this.y).getValue(),
                    Distance.quantizeDistance(this.enemyDistance).getValue(),
                    Energy.quantizeEnergy(this.enemyEnergy).getValue(),
            };
        }
        else{
            return new double[]{
                    this.x, this.y,
                    this.energy,
                    this.heading,
                    this.enemyBearing,
                    this.enemyDistance,
                    this.enemyEnergy,
            };
        }
    }


    // quantization
    private enum Energy{
        ZERO(0), CRITICAL(1), LOW(2), NORMAL(3);
        private final int value;
        Energy(int value){
            this.value = value;
        }
        public int getValue(){
            return this.value;
        }
        static Energy quantizeEnergy(double value){
            if(value == 0){
                return Energy.ZERO;
            }
            else if(value < 15){
                return Energy.CRITICAL;
            }
            else if(value < 40){
                return Energy.LOW;
            }
            else{
                return Energy.NORMAL;
            }
        }
    }

    private enum Distance{
        DANGERCLOSE(0), NEAR(1), MEDIUM(2), FAR(3);
        private final int value;
        Distance(int value){
            this.value = value;
        }
        public int getValue(){
            return this.value;
        }
        static Distance quantizeDistance(double value){
            if(value < 60){
                return Distance.DANGERCLOSE;
            }
            else if(value < 150){
                return Distance.NEAR;
            }
            else if(value < 300){
                return Distance.MEDIUM;
            }
            else{
                return Distance.FAR;
            }
        }
    }

    private enum Heading{
        NW(0), NE(1), SE(2), SW(3);
        private final int value;
        Heading(int value){this.value = value;}
        public int getValue(){return this.value;}
        static Heading quantizeHeading(double value, double x, double y){
            if(x >= 400)
                value = 360 - value;
            if(y >= 300)
                value = (540 - value) % 360;

            if(value <= 0){
                if(value > -90){
                    return Heading.NW;
                }
                else{
                    return Heading.SW;
                }
            }
            else{
                if(value <= 90){
                    return Heading.NE;
                }
                else{
                    return Heading.SE;
                }
            }
        }
    }

    private enum Bearing{
        // bearing relative to self's heading
        NW(0), NE(1), SE(2), SW(3);
        private final int value;
        Bearing(int value){this.value = value;}
        public int getValue(){return this.value;}
        static Bearing quantizeBearing(double value, double x, double y){
            if(x >= 400)
                value = 360 - value;
            if(y >= 300)
                value = 360 - value;

            if(value <= 0){
                if(value > -90){
                    return Bearing.NW;
                }
                else{
                    return Bearing.SW;
                }
            }
            else{
                if(value <= 90){
                    return Bearing.NE;
                }
                else{
                    return Bearing.SE;
                }
            }
        }
    }

    private enum Velocity {
        FORWARD(1), STILL(0), BACKWARD(-1);
        private final int value;
        Velocity(int value){this.value = value;}
        public int getValue(){return this.value;}
        static Velocity quantizeVelocity(double value) {
            if(value < -1)
                return Velocity.BACKWARD;
            else if(value > 1)
                return Velocity.FORWARD;
            else
                return Velocity.STILL;
        }
    }

    public int quantizeX(double x){
        return Math.min(7, (int) Math.abs(400 - x) / 50);
    }

    public int quantizeY(double y){
        return Math.min(5, (int) Math.abs(300 - y) / 50);
    }

    public int quantizeHeading(double angle){
        return Math.min(5, (int) Math.abs(180 - angle) / 30);
    }

    public int quantizeVelocity(double velocity, boolean enemy){
        if(!enemy)
            return Math.min(3, (int) Math.abs(velocity) / 2);
        else{
            return Math.max(-3, Math.min(3, (int) velocity / 2));
        }
    }
}
