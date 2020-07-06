CFLAGS=-I ~/research/include -std=c++11
CC=g++
ADOLC_LIB=~/research/lib64/*.dylib

stan:
	make -f ~/research/math/make/standalone stan_cos

adol: ADOL_cos.o
	$(CC) $(ADOLC_LIB) ADOL_cos.o -o adol

ADOL_cos.o: ADOL_cos.cpp
	$(CC) $(CFLAGS) -c ADOL_cos.cpp


clean:
	rm adol ADOL_cos.o ADOLC-*.tap stan_cos
