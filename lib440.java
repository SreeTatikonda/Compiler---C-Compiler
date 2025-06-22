import java.io.IOException;
import java.util.Scanner;

class lib440 {
    static Scanner in = new Scanner(System.in);

    public static int getchar() throws IOException {
        in.useDelimiter("");
        return in.next().charAt(0);
    }
    public static int putchar(int c) {
        System.out.print((char) c);
        return c;
    }
    public static int getint() {
        in.useDelimiter("[ \t\r\n]+");
        return in.nextInt();
    }
    public static void putint(int x) {
        System.out.print(x);
    }
    public static float getfloat() {
        in.useDelimiter("[ \t\r\n]+");
        return in.nextFloat();
    }
    public static void putfloat(float x) {
        System.out.printf("%.6f", x);
    }
    public static void putstring(char[] s) {
        for (int i=0; s[i]!=0; i++) {
          System.out.print(s[i]);
        }
    }
    public static char[] java2c(String s) {
        char[] chars = new char[s.length()+1];  // add one for null terminator
        s.getChars(0, s.length(), chars, 0);
        chars[s.length()] = 0;
        return chars;
    }
};
