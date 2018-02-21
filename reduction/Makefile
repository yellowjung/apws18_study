all: reduction integral integral_cal

reduction: reduction.c
	gcc -o reduction reduction.c -lm -lOpenCL

integral: integral.c
	gcc -o integral integral.c -lm -lOpenCL

integral_cal: integral_cal.c
	gcc -o integral_cal integral_cal.c -lm -lOpenCL
clean:
	rm reduction integral integral_cal
