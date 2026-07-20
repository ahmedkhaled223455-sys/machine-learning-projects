# Docker and Containers

Docker packages an application together with its dependencies into a
container image that runs identically on any machine with a container
runtime. Containers share the host kernel, making them far lighter than
virtual machines while still isolating processes and file systems.

Containers solve the "works on my machine" problem: because the image
pins the exact runtime and libraries, a pipeline behaves the same on a
laptop, in CI, and in production. A Dockerfile is the reproducible recipe
that builds the image.
