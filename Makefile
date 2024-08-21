PROGRAM := nn

CC=gcc
CFLAGS=-Wall -Wextra -O0 -I. -g -c
LDFLAGS=-lm
CFILES := $(shell find ./ -type f -name '*.c')
OBJ := $(CFILES:.c=.o)
HEADER_DEPS := $(CFILES:.c=.d)

.PHONY: all
all: $(PROGRAM)

$(PROGRAM): $(OBJ)
	@$(CC) $(OBJ) $(LDFLAGS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(PROGRAM) $(OBJ)

.PHONY: run
run: $(PROGRAM)
	@./$(PROGRAM)
