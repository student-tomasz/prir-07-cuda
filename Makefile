compiler=gcc
compile_flags=-Wall
link_flags=
sources=vec.h vec.c exitmf.h exitmf.c args.h args.c

all: cube_seq

run: all
	./cube_seq v512.dat v512_cubed.dat 1000000

bench: all
	time ./cube_seq v512.dat v512_cubed.dat 1000000 > /dev/null

clean:
	rm -rf cube_seq

cube_seq: cube_seq.c $(sources)
	$(compiler) $(compile_flags) $(link_flags) -o $@ $^
