# C-Grid Riemann Solver Mini-app

Created for the 2020 NOAA hackathon.

Line numbers refer to the source in the [2019 hackathon source](https://github.com/NOAA-GFDL/hack_2019/blob/master/gfdl_nh_core/src.v1/nh_core.F90).

## Installation

```shell
$ python3 -m venv venv
$ . venv/bin/activate
$ pip install -r requirements.txt
```

## Testing

```shell
$ python main.py Riem_solver_c_data.nc [backend]
```

Backend is an optional parameter, which defaults to `numpy` if not provided.
