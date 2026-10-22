Train 24000 / test 4800 / dense-traffic 6240 samples. Metrics in metres; MR = miss rate (final error > 2 m).

### test / all

| model | minADE@6 | minFDE@6 | MR@6 | ADE@1 | FDE@1 | MR@1 |
|---|---|---|---|---|---|---|
| constant_velocity | -- | -- | -- | 1.181 | 2.674 | 0.486 |
| constant_acceleration | -- | -- | -- | 2.780 | 7.108 | 0.935 |
| ridge | -- | -- | -- | 0.865 | 2.060 | 0.346 |
| transformer_K6 | 0.442 | 0.853 | 0.099 | 1.033 | 2.393 | 0.439 |
| transformer_K6_no_social | 0.443 | 0.865 | 0.100 | 1.038 | 2.403 | 0.423 |
| transformer_K1 | 0.853 | 2.000 | 0.329 | 0.853 | 2.000 | 0.329 |

### test / lane_change

| model | minADE@6 | minFDE@6 | MR@6 | ADE@1 | FDE@1 | MR@1 |
|---|---|---|---|---|---|---|
| constant_velocity | -- | -- | -- | 1.587 | 3.716 | 0.768 |
| constant_acceleration | -- | -- | -- | 3.107 | 8.052 | 0.957 |
| ridge | -- | -- | -- | 1.301 | 3.040 | 0.613 |
| transformer_K6 | 0.871 | 1.781 | 0.393 | 1.440 | 3.270 | 0.691 |
| transformer_K6_no_social | 0.891 | 1.809 | 0.397 | 1.476 | 3.354 | 0.671 |
| transformer_K1 | 1.305 | 3.013 | 0.590 | 1.305 | 3.013 | 0.590 |

### dense_traffic / all

| model | minADE@6 | minFDE@6 | MR@6 | ADE@1 | FDE@1 | MR@1 |
|---|---|---|---|---|---|---|
| constant_velocity | -- | -- | -- | 0.994 | 2.171 | 0.410 |
| constant_acceleration | -- | -- | -- | 2.651 | 6.748 | 0.936 |
| ridge | -- | -- | -- | 0.733 | 1.724 | 0.284 |
| transformer_K6 | 0.394 | 0.759 | 0.060 | 0.844 | 1.913 | 0.345 |
| transformer_K6_no_social | 0.401 | 0.780 | 0.051 | 0.859 | 1.920 | 0.350 |
| transformer_K1 | 0.757 | 1.708 | 0.288 | 0.757 | 1.708 | 0.288 |

### dense_traffic / lane_change

| model | minADE@6 | minFDE@6 | MR@6 | ADE@1 | FDE@1 | MR@1 |
|---|---|---|---|---|---|---|
| constant_velocity | -- | -- | -- | 1.438 | 3.363 | 0.727 |
| constant_acceleration | -- | -- | -- | 2.870 | 7.452 | 0.947 |
| ridge | -- | -- | -- | 1.146 | 2.669 | 0.547 |
| transformer_K6 | 0.827 | 1.703 | 0.377 | 1.298 | 2.930 | 0.631 |
| transformer_K6_no_social | 0.842 | 1.721 | 0.365 | 1.351 | 3.041 | 0.647 |
| transformer_K1 | 1.195 | 2.743 | 0.564 | 1.195 | 2.743 | 0.564 |
