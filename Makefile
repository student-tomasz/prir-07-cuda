cc=g++
nvcc=nvcc
ccflags=-m64
nvccflags=-m64
ldflags=-L/usr/local/cuda/lib64 -L/usr/local/cuda/lib -lcudart
deps=args.h exitmf.h vec.h
objs=args.o exitmf.o vec.o

cube_cuda.o: cube_cuda.cu $(deps)
	$(nvcc) $(nvccflags) -c -o $@ $<

%.o: %.c $(deps)
	$(cc) $(ccflags) -c -o $@ $<

cube_seq: cube_seq.o $(objs)
	$(cc) $(ccflags) -o $@ $^

cube_cuda: cube_cuda.o $(objs)
	$(cc) $(ccflags) $(ldflags) -o $@ $^

.phony: all clean bench run

all: cube_seq cube_cuda

clean:
	rm -rf cube_seq{,.o} cube_cuda{,.o} $(objs)

bench: all
	./cube_cuda v512000.dat v512000_cubed.dat 1000

run: all
	./cube_cuda v512.dat v512_cubed.dat 1
