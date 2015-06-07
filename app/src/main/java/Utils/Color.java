package Utils;

/**
 * Created by josekalladanthyil on 27/05/15.
 */
public enum Color {
    Green (android.graphics.Color.rgb(0, 255, 0)),
    Red   (android.graphics.Color.rgb(255, 0, 0)),
    Blue  (android.graphics.Color.rgb(0, 0, 255));

    private int color;
    Color(int color) {
        this.color = color;
    }

    public int getColor() {
        return color;
    }
    public static int size() {
        return values().length;
    }
}
