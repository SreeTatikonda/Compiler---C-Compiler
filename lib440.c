
#include <stdio.h>

/*
   Already in stdio.h

   int putchar(int c);

   int getchar();
*/

void putint(int x)
{
    printf("%d", x);
}

int getint()
{
    int x;
    scanf("%d", &x);
    return x;
}

void putfloat(float f)
{
    printf("%.6f", f);
}

float getfloat()
{
    float f;
    scanf("%f", &f);
    return f;
}

void putstring(const char* s)
{
    fputs(s, stdout);
}

