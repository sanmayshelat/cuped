This is a repo to describe, test, and analyse the CUPED method using simulated experiments.

* `dgp.py`: Classes with data generating processes. Data is generated in the context of the driver-side of a taxi business but it can be easily put into another context. (The distributions used do not necessarily reflect the real-world).
* `ate.py`: Traditional and CUPED methods to estimate average treatment effect of global ratio variables. Also described as "non-user metrics" or "naive means" in different sources.
* `explanation.md`: Document (put together with the help of various sources) explaining the CUPED method and relevant material.
* `sources.md`: Document listing relevant sources.