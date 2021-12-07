package Bot;
import LUT.StateActionLUT;
import NN.NNWrapper;
import robocode.*;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;

import static robocode.util.Utils.normalRelativeAngleDegrees;

public class RLBot extends AdvancedRobot{
    public enum OperationMode{
        SCAN(0), ACT(1);
        private final int value;
        OperationMode(int value){
            this.value = value;
        }
        public int getValue(){
            return this.value;
        }

        public OperationMode getType(int value) {
            for (OperationMode operationMode : OperationMode.values()) {
                if (operationMode.getValue() == value) {
                    return operationMode;
                }
            }
            return OperationMode.SCAN;
        }
    }
    private OperationMode operationMode;  // operation mode
    private final double[] previousStateActionVector = new double[State.length + 1];  // previous state action vector
    private final double[] stateActionVector = new double[State.length + 1];  // current state action vector
    private final State state = new State();  // current state
    private Action action;
    private double instantReward;
    public static RLInterface agent = new NNWrapper("out\\production\\CPEN502\\Bot\\RLBot.data\\2021-12-06-15-45-15.weights"); //new StateActionLUT(State.lowerBounds, State.upperBounds);
    private String modelFileName = getClass().getSimpleName() + "-" + agent.getClass().getSimpleName() + "-" + (new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss")).format(new Date()) + ".txt";
    static String logFileName = (new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss")).format(new Date())
            + "-" + RLBot.class.getSimpleName() + ".txt";

    // RL hypers
    public static double epsilon = 0.8;  // exploration rate, reduced over time
    private final double alpha = 0.3;  // learning rate
    private final double gamma = 0.9;  // discount factor
    private final boolean offPolicy = true;  // policy indicator

    // Counters
    public static int rounds = 0;  // # of rounds
    public static int roundTo100 = 0;
    public static int wins = 0;  // # of winning rounds
    public static int totalIterations = 0;  // # of total iterations (Q update calls)

    public void run() {
        // instructions for start, get the robot into predetermined state
        this.operationMode = OperationMode.SCAN;  // start by scanning
        this.instantReward = 0;  // clean up reward
        boolean firstAct = true;  // indicator for first action

        // decrease epsilon to 0 by 4000 rounds
        if(rounds == 5600){
            epsilon = 0;
        }

        // loop for normal behaviours
        while (true) {
            // normal behaviour when not reacting to any event
            switch (this.operationMode){
                case SCAN:{
                    // scan
                    turnRadarLeft(90);
                    break;
                }
                case ACT:{
                    // choose action
                    if(Math.random() < epsilon)
                    {
                        // explore
                        this.action = this.getRandomAction();
                    }
                    else{
                        // greedy
                        this.action = this.getGreedyAction(this.stateActionVector);
                    }
                    // act
                    this.executeAction();
                    // set state action vector
                    this.stateActionVector[this.stateActionVector.length - 1] = this.action.getValue();
                    // update Q
                    if(!firstAct) {
                        agent.updateQ(this.previousStateActionVector, computeQ(false));
                    }
                    else{
                        // do not update Q after the very first action
                        firstAct = false;
                    }
                    this.operationMode = OperationMode.SCAN;
                    break;
                }
            }
            // reverse operation
//            this.operationMode = this.operationMode.getType(1 - this.operationMode.getValue());
        }
    }

    /**
     * Get randomized action
     * @return randomized action
     */
    private Action getRandomAction(){
        return Action.getAction((int) System.currentTimeMillis() % Action.values().length);
    }

    /**
     * Get greedy action by selecting the action such that Q(s, a) is maximum
     * @param stateActionVector current state action vector containing current state
     * @return greedy action
     */
    private Action getGreedyAction(double[] stateActionVector){
        double maxQ = - Double.MAX_VALUE;
        Action greedyAction = Action.getAction(0);  // default
        for(Action a: Action.values()){
            // find a such that Q(s, a) is the largest
            stateActionVector[stateActionVector.length - 1] = a.getValue();
            double curQ = agent.forward(stateActionVector)[0];
            if(curQ > maxQ){
                greedyAction = a;
                maxQ = curQ;
            }
        }
        stateActionVector[stateActionVector.length - 1] = greedyAction.getValue();
        return greedyAction;
    }

    /**
     * Compute the new Q(s, a) value based on r and s'
     * @param terminal indicator for terminal states
     * @return new value of Q(s, a)
     */
    private double computeQ(boolean terminal){
        double[] targetStateActionVector;
        if(!terminal){
            if(this.offPolicy){
                // use greedy action
                targetStateActionVector = new double[this.stateActionVector.length];
                System.arraycopy(this.stateActionVector, 0, targetStateActionVector, 0, State.length);
                this.getGreedyAction(targetStateActionVector);
            }
            else{
                // use current action
                targetStateActionVector = this.stateActionVector;
            }
        }
        else{
            targetStateActionVector = this.stateActionVector;
            // set action value for terminal state as 0
            targetStateActionVector[targetStateActionVector.length - 1] = 0;
        }
        double prevQ = agent.forward(this.previousStateActionVector)[0];
        this.instantReward = Math.max(-1, Math.min(1, this.instantReward));
        double newQ = prevQ + this.alpha * (this.instantReward +
                this.gamma * agent.forward(targetStateActionVector)[0] - prevQ);
        // clear intermediate reward
        this.instantReward = 0;
        // update count
        totalIterations += 1;
        // reduce epsilon after certain # of iterations
//        if(totalIterations % 5000 == 0){
//            epsilon = Math.max(0, epsilon - 0.05);
//            out.println("[INFO] New exploration rate = " + epsilon);
//        }
        return newQ;
    }

    /**
     * Execute selected action
     */
    private void executeAction(){
//        out.println(this.action.name());
        switch (this.action){
            case ADVANCE:
            {
                setTurnRight(this.state.getEnemyBearing());
                setAhead(100);
                execute();
            }
            break;
            case RETREAT:
            {
                if(this.state.getEnemyBearing() > 0)
                {
                    setTurnLeft(100 - this.state.getEnemyBearing());
                }
                else{
                    setTurnRight(100 + this.state.getEnemyBearing());
                }
                setAhead(100);
                execute();
            }
            break;
            case FIRE:
            {
                // turn gun at enemy
                turnGunRight(normalRelativeAngleDegrees(
                        getHeading() - getGunHeading() + this.state.getEnemyBearing()));
                fire(2);
            }
            break;
            case CIRCLE:
            {
                if(this.state.getEnemyBearing() > 0){
                    setTurnRight(this.state.getEnemyBearing() - 90);
                }
                else{
                    setTurnRight(this.state.getEnemyBearing() + 90);
                }
                setAhead(50);
                execute();
            }
            break;
            case HEAD2CENTER:
            {
                double[] toCenter = new double[]{400 - this.getX(), 300 - this.getY()};
                double lengthToCenter = Math.sqrt(Math.pow(toCenter[0], 2) + Math.pow(toCenter[1], 2));
                toCenter[0] /= lengthToCenter;
                toCenter[1] /= lengthToCenter;
//                double[] heading = new double[]{Math.sin(getHeadingRadians()), Math.cos(getHeadingRadians())};
                double cosCenter = Math.min(1., Math.max(-1., 0 * toCenter[0] + 1 * toCenter[1]));
                double centerRadian = Math.acos(cosCenter);
                if(toCenter[0] < 0)
                    centerRadian = 2 * Math.PI - centerRadian;
                setTurnRightRadians(centerRadian - getHeadingRadians());
                setAhead(100);
                execute();
            }
            break;
        }
    }

    public void onScannedRobot(ScannedRobotEvent e) {
        // store previous state
        System.arraycopy(this.stateActionVector, 0, this.previousStateActionVector, 0, State.length + 1);
        // capture state
        this.state.setMyState(getX(), getY(), getEnergy(), getHeading(), getGunHeading(), getVelocity());
        this.state.setEnemyState(e.getBearing(), e.getDistance(), e.getEnergy(), e.getHeading(), e.getVelocity());
        // update state
        System.arraycopy(this.state.getStateVector(), 0, this.stateActionVector, 0, State.length);
        // reverse operation mode
        this.operationMode = OperationMode.ACT;
    }

    // Receive reward based on event
    public void onBulletHit(BulletHitEvent e){
        // 3x fire power points
        this.instantReward += Reward.BULLETHIT.getValue();// * e.getBullet().getPower();
    }

    public void onBulletMissed(BulletMissedEvent e){
        // negative fire power points
        this.instantReward += Reward.BULLETMISSED.getValue();// * e.getBullet().getPower();
    }

    public void onHitByBullet(HitByBulletEvent e){
        this.instantReward += Reward.HITBYBULLET.getValue();// * e.getPower();
    }

    public void onHitRobot(HitRobotEvent e){
        if(e.isMyFault()){
            // 1.2 points
            this.instantReward += Reward.HITROBOT.getValue();
        }
        else{
            // -0.6 points
            this.instantReward += Reward.HITBYROBOT.getValue();
        }
    }

    public void onHitWall(HitWallEvent event){
        this.instantReward += Reward.HITWALL.getValue();
    }

    // update Q when game ends
    public void onDeath(DeathEvent e){
        // receive terminal reward
        this.instantReward += Reward.DEATH.getValue();
    }

    public void onWin(WinEvent event){
        // receive terminal reward
        this.instantReward += Reward.WIN.getValue();

        // increment win counter
        wins += 1;
    }

    private void printMatchStatistics(){
        double winRate = (double) wins / roundTo100;
        // print
        out.println("[INFO] Matches " + rounds + "/120000, W/L " + winRate);
        out.println("[INFO] Total iterations: " + totalIterations);
        // log
        PrintStream log = null;
        try{
            log = new PrintStream(new RobocodeFileOutputStream(getDataDirectory() + "\\" + logFileName, true));
        } catch (IOException e) {
            System.out.println("[ERROR] Unable to create log file.");
            return;
        }
        log.println(rounds + " " + String.format("%.2f", winRate));
        log.flush();
        log.close();
    }

    public void onRoundEnded(RoundEndedEvent event){
        // update Q
        agent.updateQ(this.previousStateActionVector, computeQ(true));

        // update statistics
        rounds += 1;
        roundTo100 += 1;
        if(roundTo100 == 200) {
            this.printMatchStatistics();
            roundTo100 = 0;
            wins = 0;
        }
    }

    public void onBattleEnded(BattleEndedEvent event){
        // save model
        agent.save(new File(getDataDirectory() + "\\" + this.modelFileName));
    }
}
