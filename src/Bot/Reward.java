package Bot;

public enum Reward {
    BULLETHIT(2./4), BULLETMISSED(-1.5/4), HITBYBULLET(-2./4),
    HITROBOT(1.2/4), HITBYROBOT(-0.6/4),
    HITWALL(-0.5/4), DEATH(-1.0), WIN(1.0);
    private final double value;
    Reward(double value){
        this.value = value;
    }
    public double getValue(){
        return this.value;
    }
}
