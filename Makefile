LIBS = $(shell pkg-config --cflags --libs /usr/local/Cellar/opencv/4.2.0_3/lib/pkgconfig/opencv4.pc)

debug:
	g++ $(LIBS) -std=c++11 main.cpp -o test