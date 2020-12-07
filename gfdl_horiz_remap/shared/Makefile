# ------------------------------------------------------------------------------
# FMS/FRE Project: Makefile to build regridding executables
# ------------------------------------------------------------------------------
# Copyright (C) NOAA Geophysical Fluid Dynamics Laboratory, 2009-2010, 2017
# Designed and written by V. Balaji, Amy Langenhorst, Aleksey Yakovlev and
# Seth Underwood
# ------------------------------------------------------------------------------

# MPICC and CC default settings
MPICC    := nvc
CC       := nvc

# Get the site specific settings, CC and MPICC may be overridden
include ../nvhpc.mk

# Default Directories
SRCDIR := .
PREFIX := .
PREFIX_EXEC = $(PREFIX)/$(SITE)/bin
PREFIX_SHARE = $(PREFIX)/share
PREFIX_DOCS = $(PREFIX_SHARE)
PREFIX_SHARE_EXEC = $(PREFIX_SHARE)/bin

VPATH = $(SRCDIR):.

CFLAGS   := -O3 -g -traceback $(CFLAGS_SITE)
CFLAGS_O2:= -O2 -g -traceback $(CFLAGS_SITE)
INCLUDES := -I${NETCDF_HOME}/include -I$(SRCDIR)/ -I$(SRCDIR)/../shared -I$(SRCDIR)/../../shared/mosaic

TARGETS  := all
TARGETS_PARALLEL := all
ifndef NOPARALLEL
TARGETS += $(TARGETS_PARALLEL)
endif

SOURCES  += mosaic_util.c mpp.c
SOURCES  += create_xgrid.c interp.c read_mosaic.c
SOURCES  += mpp_domain.c mpp_io.c tool_util.c affinity.c

OBJECTS  := $(SOURCES:c=o)

HEADERS = mpp.h mpp_domain.h mpp_io.h tool_util.h   \
          constant.h create_xgrid.h interp.h  \
          mosaic_util.h read_mosaic.h

all: $(OBJECTS)

mosaic_util.o: mosaic_util.c $(HEADERS)
	$(CC) $(CFLAGS) $(INCLUDES) -c $<

mosaic_util_parallel.o: mosaic_util.c $(HEADERS)
	$(MPICC) -Duse_libMPI $(CFLAGS) $(INCLUDES) -o $@ -c $<

mpp.o: mpp.c $(HEADERS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

mpp_parallel.o: mpp.c $(HEADERS)
	$(MPICC) -Duse_libMPI $(CFLAGS) $(INCLUDES) -o $@ -c $<

create_xgrid.o: create_xgrid.c $(HEADERS)
	$(CC) $(CFLAGS_O2) $(INCLUDES) -c $<

interp.o: interp.c $(HEADERS)
	$(CC) $(CFLAGS) $(INCLUDES) -c $<

read_mosaic.o: read_mosaic.c $(HEADERS)
	$(CC) -Duse_netCDF $(CFLAGS) $(INCLUDES) -c $<

mpp_domain.o: mpp_domain.c $(HEADERS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

mpp_io.o: mpp_io.c $(HEADERS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

tool_util.o: tool_util.c $(HEADERS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

affinity.o: affinity.c $(HEADERS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $<

install: all
	install -m 755 -d $(PREFIX)
	install -m 755 -d $(PREFIX_EXEC)
	install -m 755 $(TARGETS) $(PREFIX_EXEC)

docs:

install-docs: docs

clean:
	-rm -f *.o $(TARGETS)
