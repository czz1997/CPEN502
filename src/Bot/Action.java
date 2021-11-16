package Bot;

public enum Action {
    CIRCLE(2), HEAD2CENTER(4),
    ADVANCE(0), RETREAT(1),
    FIRE(3);
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
        return Action.ADVANCE;
    }
}
