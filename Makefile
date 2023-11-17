all:
	(cd openmp ; make all)
	(cd cuda ; make all)

clean:
	(cd openmp ; make clean)
	(cd cuda ; make clean)