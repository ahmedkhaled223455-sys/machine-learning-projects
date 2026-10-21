## Verification

- Terzaghi consolidation: max error 1.7e-05 (of u0) on the finest grid; observed order 2.01, 2.00, 2.00
- Radial injection solver vs Theis (max relative error over 3 reservoirs x 3 radii, 10 years):

| cells | steps | max rel. error |
|---|---|---|
| 60 | 75 | 0.808% |
| 120 | 150 | 0.502% |
| 240 | 300 | 0.097% |
| 480 | 600 | 0.017% |

## Poroelastic coupling

- 4000 random faults; 4.4% already critically stressed
- median critical pressure increase: 11.9 MPa (pressure only) vs 30.7 MPa with the stress path (median ratio 2.63)
- fault slip (not hydraulic fracturing) is the binding limit for 38% of faults

## Safe-rate surrogates (log10 m3/day)

Samples: train 1916, test 562, out-of-range 560; simulation time 16 s.

### test_in_range

| model | RMSE (log10) | median abs. error | unsafe over-prediction (>10%) |
|---|---|---|---|
| black_box_ridge | 0.049 | 5.3% | 9.1% |
| physics_only | 0.001 | 0.2% | 0.0% |
| physics_hybrid | 0.001 | 0.1% | 0.0% |

### test_out_of_range_k

| model | RMSE (log10) | median abs. error | unsafe over-prediction (>10%) |
|---|---|---|---|
| black_box_ridge | 0.125 | 8.8% | 25.7% |
| physics_only | 0.001 | 0.1% | 0.0% |
| physics_hybrid | 0.000 | 0.1% | 0.0% |

## Reduced-order model (POD + RBF)

- 8 POD modes capture 99.9917% of snapshot energy
- pressure-profile error: 0.15% (truncation only), 0.20% median / 1.15% p95 (predicted)
- 115 microseconds per profile

## Monte Carlo fault-reactivation risk

- 2000 realisations of uncertain permeability, friction, and minimum stress; 0.1% already critical
- rate at 5% risk: 844 m3/day (simulator) vs 842 m3/day (hybrid surrogate)
- simulator 11.3 s vs surrogate 2.5 ms (4573x faster)
