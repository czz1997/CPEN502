package Bot;

public enum Reward {
    BULLETHIT(0.5), BULLETMISSED(-0.25), HITBYBULLET(-0.5),
    HITROBOT(0.25/4), HITBYROBOT(-0.25/4),
    HITWALL(-0.1/4), DEATH(-1.0), WIN(1.0);
    private final double value;
    Reward(double value){
        this.value = value;
    }
    public double getValue(){
        return this.value;
    }
}
