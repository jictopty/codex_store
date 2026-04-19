CC ?= gcc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -pedantic
LDFLAGS ?= -lm
TARGET ?= ns_dg_solver

.PHONY: all run test clean mpi

all: $(TARGET)

$(TARGET): main.c
	$(CC) $(CFLAGS) main.c -o $(TARGET) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) 0

test: $(TARGET)
	bash tests/test_case.sh ./$(TARGET)

mpi:
	mpicc $(CFLAGS) -DUSE_MPI main.c -o $(TARGET)_mpi $(LDFLAGS)

clean:
	rm -f $(TARGET) $(TARGET)_mpi
