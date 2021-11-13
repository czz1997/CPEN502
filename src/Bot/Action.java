package Bot;

public enum Action {
    CIRCLE(0), HEAD2CENTER(1),
    ADVANCE(2), RETREAT(3),
    FIRE(4);
    // STOP, RETREAT, ADVANCE, CIRCLE;
    private final int value;
    Action(int value){
        this.value = value;
    }
    public int getValue(){
        return this.value;
    }
    static Action getAction(int value){
        for(Action a: Action.values()){
            if(value == a.getValue()){
                return a;
            }
        }
        return Action.CIRCLE;
    }
}
