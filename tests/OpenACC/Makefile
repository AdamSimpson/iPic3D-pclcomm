all: clean checkEnv setFlags alloc

.PHONY: checkEnv setFlags clean

checkEnv:
ifndef CRAY_CUDATOOLKIT_VERSION
    $(error cudatoolkit module not loaded)
endif

setFlags:
ifdef CRAY_PRGENVPGI
        CPPFLAGS = -acc -Minfo=acc
endif

alloc:	test_alloc.cpp
	CC $(CPPFLAGS) -o alloc test_alloc.cpp -I../../include/

clean:
	rm -f *.o
	rm -f alloc
